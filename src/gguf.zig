const Self = @This();

allocator: ?std.mem.Allocator = null,

meta: Meta,
config_map: ConfigMap,
vocabulary: Vocabulary,
vocabulary_ptr: *Vocabulary,
tensor_infos: TensorInfoMap,
file: std.fs.File,
data_offset: u64,

const TensorInfoMap = std.StringHashMapUnmanaged(TensorInfo);

const TensorInfo = struct {
    n_dimensions: u32,
    dimensions: [4]u64,
    gguf_type: u32,
    offset: u64,

    fn dataType(self: TensorInfo) !DataType {
        return switch (self.gguf_type) {
            0 => .FP32,
            1 => .FP16,
            2 => .Q4_0,
            3 => .Q4_1,
            6 => .Q5_0,
            8 => .Q8_0,
            12 => .Q4_K,
            13 => .Q5_K,
            14 => .Q6_K,
            30 => .BF16,
            else => {
                log.err("unsupported gguf tensor type: {d}", .{self.gguf_type});
                return error.IOError;
            },
        };
    }

    fn byteSize(self: TensorInfo) !u64 {
        const dtype = try self.dataType();
        var count: u64 = 1;
        for (self.dimensions[0..self.n_dimensions]) |d| {
            if (d == 0) return error.IOError;
            count = std.math.mul(u64, count, d) catch return error.IOError;
        }
        if (count > std.math.maxInt(usize)) return error.IOError;
        return dtype.byteSize(@intCast(count)) catch return error.IOError;
    }
};

// -- GGUF metadata value types --

const MetadataValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    uint64: u64,
    int64: i64,
    float64: f64,
    bool_val: bool,
    string: []const u8,
    array: ArrayValue,

    const ArrayValue = struct {
        elem_type: u32,
        values: []const MetadataValue,
    };

    fn getUint(self: MetadataValue) ?u64 {
        return switch (self) {
            .uint8 => |value| value,
            .uint16 => |value| value,
            .uint32 => |value| value,
            .uint64 => |value| value,
            .int8 => |value| if (value >= 0) @intCast(value) else null,
            .int16 => |value| if (value >= 0) @intCast(value) else null,
            .int32 => |value| if (value >= 0) @intCast(value) else null,
            .int64 => |value| if (value >= 0) @intCast(value) else null,
            else => null,
        };
    }

    fn getFloat(self: MetadataValue) ?f64 {
        return switch (self) {
            .float32 => |value| value,
            .float64 => |value| value,
            .uint8 => |value| @floatFromInt(value),
            .uint16 => |value| @floatFromInt(value),
            .uint32 => |value| @floatFromInt(value),
            .int8 => |value| @floatFromInt(value),
            .int16 => |value| @floatFromInt(value),
            .int32 => |value| @floatFromInt(value),
            else => null,
        };
    }

    fn getString(self: MetadataValue) ?[]const u8 {
        return switch (self) {
            .string => |value| value,
            else => null,
        };
    }
};

const MetadataMap = std.StringHashMapUnmanaged(MetadataValue);

// -- GGUF binary reading helpers --

fn readU32(file: std.fs.File) !u32 {
    var buf: [4]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 4) return error.ReaderError;
    return std.mem.readInt(u32, &buf, .little);
}

fn readU64(file: std.fs.File) !u64 {
    var buf: [8]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 8) return error.ReaderError;
    return std.mem.readInt(u64, &buf, .little);
}

fn readI8(file: std.fs.File) !i8 {
    var buf: [1]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 1) return error.ReaderError;
    return @bitCast(buf[0]);
}

fn readU8(file: std.fs.File) !u8 {
    var buf: [1]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 1) return error.ReaderError;
    return buf[0];
}

fn readI16(file: std.fs.File) !i16 {
    var buf: [2]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 2) return error.ReaderError;
    return std.mem.readInt(i16, &buf, .little);
}

fn readU16(file: std.fs.File) !u16 {
    var buf: [2]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 2) return error.ReaderError;
    return std.mem.readInt(u16, &buf, .little);
}

fn readI32(file: std.fs.File) !i32 {
    var buf: [4]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 4) return error.ReaderError;
    return std.mem.readInt(i32, &buf, .little);
}

fn readI64(file: std.fs.File) !i64 {
    var buf: [8]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 8) return error.ReaderError;
    return std.mem.readInt(i64, &buf, .little);
}

fn readF32(file: std.fs.File) !f32 {
    var buf: [4]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 4) return error.ReaderError;
    return @bitCast(std.mem.readInt(u32, &buf, .little));
}

fn readF64(file: std.fs.File) !f64 {
    var buf: [8]u8 = undefined;
    const bytes_read = try file.readAll(&buf);
    if (bytes_read != 8) return error.ReaderError;
    return @bitCast(std.mem.readInt(u64, &buf, .little));
}

fn readBool(file: std.fs.File) !bool {
    return (try readU8(file)) != 0;
}

fn readString(allocator: std.mem.Allocator, file: std.fs.File) ![]const u8 {
    const len = try readU64(file);
    if (len > 1024 * 1024) {
        log.err("gguf: string length {d} exceeds sanity limit", .{len});
        return error.ReaderError;
    }
    const buf = try allocator.alloc(u8, @intCast(len));
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != @as(usize, @intCast(len))) {
        log.err("gguf: string read incomplete: got {d}/{d} bytes", .{ bytes_read, len });
        return error.ReaderError;
    }
    return buf;
}

fn readMetadataValue(allocator: std.mem.Allocator, file: std.fs.File, value_type: u32, depth: u32) !MetadataValue {
    if (depth > 4) {
        log.err("gguf: metadata recursion depth exceeds limit", .{});
        return error.ReaderError;
    }
    return switch (value_type) {
        0 => .{ .uint8 = try readU8(file) }, // UINT8
        1 => .{ .int8 = try readI8(file) }, // INT8
        2 => .{ .uint16 = try readU16(file) }, // UINT16
        3 => .{ .int16 = try readI16(file) }, // INT16
        4 => .{ .uint32 = try readU32(file) }, // UINT32
        5 => .{ .int32 = try readI32(file) }, // INT32
        6 => .{ .float32 = try readF32(file) }, // FLOAT32
        7 => .{ .bool_val = try readBool(file) }, // BOOL
        8 => .{ .string = try readString(allocator, file) }, // STRING
        9 => blk: { // ARRAY
            const elem_type = try readU32(file);
            const count = try readU64(file);
            if (count > 10 * 1024 * 1024) {
                log.err("gguf: array count {d} exceeds sanity limit", .{count});
                return error.ReaderError;
            }
            const values = try allocator.alloc(MetadataValue, @intCast(count));
            errdefer allocator.free(values);
            for (0..@intCast(count)) |index| {
                values[index] = try readMetadataValue(allocator, file, elem_type, depth + 1);
            }
            break :blk .{ .array = .{ .elem_type = elem_type, .values = values } };
        },
        10 => .{ .uint64 = try readU64(file) }, // UINT64
        11 => .{ .int64 = try readI64(file) }, // INT64
        12 => .{ .float64 = try readF64(file) }, // FLOAT64
        else => {
            log.err("gguf: unsupported metadata value type: {d}", .{value_type});
            return error.ReaderError;
        },
    };
}

fn freeMetadataValue(allocator: std.mem.Allocator, value: MetadataValue) void {
    switch (value) {
        .string => |str| allocator.free(str),
        .array => |arr| {
            for (arr.values) |element| freeMetadataValue(allocator, element);
            allocator.free(arr.values);
        },
        else => {},
    }
}

// -- Parsing the GGUF file --

fn parseGguf(allocator: std.mem.Allocator, file: std.fs.File) !struct {
    metadata: MetadataMap,
    tensor_infos: TensorInfoMap,
    data_offset: u64,
} {
    // Read and validate magic
    const magic = try readU32(file);
    const gguf_magic = 0x46554747;
    if (magic != gguf_magic) {
        log.err("gguf: invalid magic: 0x{x} (expected 0x{x})", .{ magic, gguf_magic });
        return error.ReaderError;
    }

    // Read and validate version
    const version = try readU32(file);
    const supported_version = 3;
    if (version != supported_version) {
        log.err("gguf: unsupported version: {d} (expected {d})", .{ version, supported_version });
        return error.ReaderError;
    }

    // Read counts
    const tensor_count = try readU64(file);
    const metadata_kv_count = try readU64(file);

    // Read metadata KV pairs
    var metadata: MetadataMap = .empty;
    errdefer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            freeMetadataValue(allocator, entry.value_ptr.*);
        }
        metadata.deinit(allocator);
    }

    for (0..@intCast(metadata_kv_count)) |_| {
        const key = try readString(allocator, file);
        errdefer allocator.free(key);
        const value_type = try readU32(file);
        const value = try readMetadataValue(allocator, file, value_type, 0);
        try metadata.put(allocator, key, value);
    }

    // Read tensor info entries
    var tensor_infos: TensorInfoMap = .empty;
    errdefer {
        var it = tensor_infos.iterator();
        while (it.next()) |entry| allocator.free(entry.key_ptr.*);
        tensor_infos.deinit(allocator);
    }

    for (0..@intCast(tensor_count)) |_| {
        const name = try readString(allocator, file);
        errdefer allocator.free(name);

        const n_dimensions = try readU32(file);
        if (n_dimensions > 4) {
            log.err("gguf: tensor has {d} dimensions (max 4)", .{n_dimensions});
            return error.ReaderError;
        }

        var dimensions: [4]u64 = .{ 0, 0, 0, 0 };
        for (0..n_dimensions) |dim_index| {
            dimensions[dim_index] = try readU64(file);
        }

        const gguf_type = try readU32(file);
        const offset = try readU64(file);

        try tensor_infos.put(allocator, name, .{
            .n_dimensions = n_dimensions,
            .dimensions = dimensions,
            .gguf_type = gguf_type,
            .offset = offset,
        });
    }

    // Calculate data section offset (align to alignment, default 32)
    const default_alignment = 32;
    const alignment: u64 = blk: {
        if (metadata.get("general.alignment")) |val| {
            if (val.getUint()) |align_value| break :blk align_value;
        }
        break :blk default_alignment;
    };

    const current_pos = try file.getPos();
    const data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

    return .{
        .metadata = metadata,
        .tensor_infos = tensor_infos,
        .data_offset = data_offset,
    };
}

// -- ConfigMap extraction from metadata --

fn extractConfigMap(allocator: std.mem.Allocator, metadata: *const MetadataMap) !struct {
    config_map: ConfigMap,
    meta: Meta,
} {
    // Get architecture name
    const arch = blk: {
        if (metadata.get("general.architecture")) |val| {
            if (val.getString()) |s| break :blk s;
        }
        log.err("gguf: missing 'general.architecture' metadata", .{});
        return error.ReaderError;
    };

    // Build architectures for Meta (raw GGUF name, no mapping)
    const architectures = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(architectures);
    architectures[0] = try allocator.dupe(u8, arch);
    errdefer allocator.free(architectures[0]);

    // Build prefix to strip
    const arch_prefix_len = arch.len + 1; // "llama."

    // Build ConfigMap
    var config_map: ConfigMap = .empty;
    errdefer deinitConfigMap(&config_map, allocator);

    var it = metadata.iterator();
    while (it.next()) |entry| {
        const raw_key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        // Skip general.architecture (goes to Meta)
        if (std.mem.eql(u8, raw_key, "general.architecture")) continue;

        // Determine the key to use in ConfigMap
        const map_key = if (std.mem.startsWith(u8, raw_key, arch) and
            raw_key.len > arch.len and raw_key[arch.len] == '.')
            // Strip arch prefix: "llama.block_count" -> "block_count"
            try allocator.dupe(u8, raw_key[arch_prefix_len..])
        else
            // Global key: keep as-is
            try allocator.dupe(u8, raw_key);
        errdefer allocator.free(map_key);

        // Convert MetadataValue to ConfigValue
        const config_value: ?ConfigValue = switch (value) {
            .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64 => blk: {
                if (value.getUint()) |uint_val| break :blk .{ .uint = @intCast(uint_val) };
                break :blk null;
            },
            .float32, .float64 => blk: {
                if (value.getFloat()) |float_val| break :blk .{ .float = @floatCast(float_val) };
                break :blk null;
            },
            .bool_val => |bool_value| .{ .boolean = bool_value },
            .string => |str| blk: {
                const duped = try allocator.dupe(u8, str);
                break :blk .{ .string = duped };
            },
            .array => |arr| blk: {
                if (arr.values.len == 0) break :blk null;

                // Check if it's a uint array
                const first = arr.values[0];
                if (first.getUint() != null) {
                    const indices = try allocator.alloc(usize, arr.values.len);
                    var valid = true;
                    for (arr.values, 0..) |elem, elem_index| {
                        if (elem.getUint()) |uint_val| {
                            indices[elem_index] = @intCast(uint_val);
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        break :blk .{ .uint_array = indices };
                    }
                    allocator.free(indices);
                }
                // Check if it's a string array
                if (first.getString() != null) {
                    const strs = try allocator.alloc([]const u8, arr.values.len);
                    var loaded: usize = 0;
                    errdefer {
                        for (strs[0..loaded]) |str| allocator.free(str);
                        allocator.free(strs);
                    }
                    var valid = true;
                    for (arr.values, 0..) |elem, elem_index| {
                        if (elem.getString()) |str| {
                            strs[elem_index] = try allocator.dupe(u8, str);
                            loaded += 1;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        break :blk .{ .string_array = strs };
                    }
                    for (strs[0..loaded]) |str| allocator.free(str);
                    allocator.free(strs);
                }
                break :blk null;
            },
        };

        if (config_value) |cv| {
            try config_map.put(allocator, map_key, cv);
        } else {
            allocator.free(map_key);
        }
    }

    // Vocabulary size from tokens array length
    const vocabulary_size: usize = blk: {
        if (metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val == .array) break :blk val.array.values.len;
        }
        break :blk configGetUint(config_map, "vocab_size") orelse 0;
    };

    const meta: Meta = .{
        .architectures = architectures,
        .data_types = .none, // Populated later from tensor types
        .vocabulary_size = vocabulary_size,
        .max_len = configGetUint(config_map, "context_length") orelse 2048,
        .bos_token_id = configGetUint(config_map, "tokenizer.ggml.bos_token_id") orelse 1,
        .eos_token_id = configGetUint(config_map, "tokenizer.ggml.eos_token_id") orelse 2,
    };

    return .{
        .config_map = config_map,
        .meta = meta,
    };
}

fn getMetadataGlobalUsize(metadata: *const MetadataMap, key: []const u8) ?usize {
    if (metadata.get(key)) |val| {
        if (val.getUint()) |uint_val| return @intCast(uint_val);
    }
    return null;
}

// -- Vocabulary extraction from metadata --

fn extractVocabulary(allocator: std.mem.Allocator, metadata: *const MetadataMap) !struct {
    vocabulary_ptr: *Vocabulary,
    vocabulary: Vocabulary,
} {
    // Get tokens array
    const tokens_val = metadata.get("tokenizer.ggml.tokens") orelse {
        log.err("gguf: missing 'tokenizer.ggml.tokens' metadata", .{});
        return error.ReaderError;
    };
    if (tokens_val != .array) {
        log.err("gguf: 'tokenizer.ggml.tokens' is not an array", .{});
        return error.ReaderError;
    }
    const tokens = tokens_val.array.values;

    // Get optional token types
    const token_types: ?[]const MetadataValue = blk: {
        if (metadata.get("tokenizer.ggml.token_type")) |val| {
            if (val == .array) break :blk val.array.values;
        }
        break :blk null;
    };

    // Build encoding and decoding maps
    var encoding: Vocabulary.EncodingVocabulary = .empty;
    errdefer {
        var enc_it = encoding.iterator();
        while (enc_it.next()) |entry| allocator.free(entry.key_ptr.*);
        encoding.deinit(allocator);
    }
    var decoding: Vocabulary.DecodingVocabulary = .empty;
    errdefer {
        var dec_it = decoding.iterator();
        while (dec_it.next()) |entry| allocator.free(entry.value_ptr.*);
        decoding.deinit(allocator);
    }

    var unknown_token: ?[]const u8 = null;

    // GGUF token_type 3 = CONTROL (e.g. <|im_start|>, <|im_end|>),
    // type 4 = USER_DEFINED. Both must land in `special_tokens` so the base
    // tokenizer's `encode` splits them out before BPE; otherwise chat-template
    // markers get BPE'd into ASCII pieces (`<` `|` `im` `_start` `|` `>`),
    // turn boundaries are invisible, and the model never emits EOT.
    var special_tokens: Vocabulary.SpecialTokens = .empty;
    errdefer {
        var sp_it = special_tokens.iterator();
        while (sp_it.next()) |entry| allocator.free(entry.key_ptr.*);
        special_tokens.deinit(allocator);
    }

    for (tokens, 0..) |token_val, idx| {
        const token_str = token_val.getString() orelse continue;
        const token_id: Vocabulary.TokenID = @intCast(idx);

        const enc_key = try allocator.dupe(u8, token_str);
        errdefer allocator.free(enc_key);
        const dec_val = try allocator.dupe(u8, token_str);
        errdefer allocator.free(dec_val);

        try encoding.put(allocator, enc_key, token_id);
        try decoding.put(allocator, token_id, dec_val);

        if (token_types) |types| {
            if (idx < types.len) {
                if (types[idx].getUint()) |token_type| {
                    switch (token_type) {
                        2 => unknown_token = dec_val, // UNKNOWN
                        3, 4 => { // CONTROL or USER_DEFINED
                            const sp_key = try allocator.dupe(u8, token_str);
                            errdefer allocator.free(sp_key);
                            try special_tokens.put(allocator, sp_key, token_id);
                        },
                        else => {},
                    }
                }
            }
        }
    }

    // Build sorted-by-length-descending list so longer tokens match first
    // (e.g. "<|im_start|>" before "<|").
    var special_sorted = try allocator.alloc(Vocabulary.SpecialTokenEntry, special_tokens.count());
    errdefer allocator.free(special_sorted);
    {
        var i: usize = 0;
        var sp_it = special_tokens.iterator();
        while (sp_it.next()) |entry| : (i += 1) {
            special_sorted[i] = .{ .text = entry.key_ptr.*, .id = entry.value_ptr.* };
        }
        std.mem.sort(Vocabulary.SpecialTokenEntry, special_sorted, {}, struct {
            fn lessThan(_: void, a: Vocabulary.SpecialTokenEntry, b: Vocabulary.SpecialTokenEntry) bool {
                return a.text.len > b.text.len;
            }
        }.lessThan);
    }

    // Build merge index
    var merge_index: Vocabulary.MergePairIndex = .empty;
    errdefer {
        var merge_it = merge_index.iterator();
        while (merge_it.next()) |entry| allocator.free(entry.key_ptr.*);
        merge_index.deinit(allocator);
    }

    if (metadata.get("tokenizer.ggml.merges")) |merges_val| {
        if (merges_val == .array) {
            var key_buf: std.ArrayListUnmanaged(u8) = .{};
            defer key_buf.deinit(allocator);

            for (merges_val.array.values, 0..) |merge_val, idx| {
                const merge_str = merge_val.getString() orelse continue;
                const split = std.mem.indexOfScalar(u8, merge_str, ' ') orelse continue;
                const first = merge_str[0..split];
                const second = merge_str[split + 1 ..];

                key_buf.clearRetainingCapacity();
                try key_buf.appendSlice(allocator, first);
                try key_buf.append(allocator, 0);
                try key_buf.appendSlice(allocator, second);

                const key = try allocator.dupe(u8, key_buf.items);
                errdefer allocator.free(key);
                try merge_index.put(allocator, key, idx);
            }
        }
    }

    // Determine byte-level tokenizer
    const use_byte_level = blk: {
        if (metadata.get("tokenizer.ggml.model")) |val| {
            if (val.getString()) |s| {
                break :blk std.mem.eql(u8, s, "gpt2");
            }
        }
        break :blk false;
    };

    // Build post-processor for BOS token
    const add_bos = blk: {
        if (metadata.get("tokenizer.ggml.add_bos_token")) |val| {
            break :blk val == .bool_val and val.bool_val;
        }
        break :blk true; // default: add BOS
    };

    const post_processor: ?Vocabulary.PostProcessor = if (add_bos) blk: {
        const bos_id: Vocabulary.TokenID = @intCast(
            getMetadataGlobalUsize(metadata, "tokenizer.ggml.bos_token_id") orelse 1,
        );
        const template = try allocator.alloc(Vocabulary.PostProcessor.TemplateProcessing, 2);
        template[0] = .{ .special_token = bos_id };
        template[1] = .{ .sequence = {} };
        break :blk .{ .template = template };
    } else null;

    const vocab_ptr = try allocator.create(Vocabulary);
    vocab_ptr.* = .{
        .encoding = encoding,
        .decoding = decoding,
        .merge_index = merge_index,
        .unknown_token = unknown_token,
        .use_byte_level = use_byte_level,
        .post_processor = post_processor,
        .special_tokens = special_tokens,
        .special_tokens_sorted = special_sorted,
    };

    return .{
        .vocabulary_ptr = vocab_ptr,
        .vocabulary = vocab_ptr.*,
    };
}

// -- Public tensor reading --

/// Parser-native data-type enum. Exactly mirrors the numeric values of
/// `base.Tensor.DataType` so conversions between the two are a single
/// `@enumFromInt(@intFromEnum(...))` cast (used by `harness.adapters`).
/// The parser no longer depends on base — callers route through
/// `getTensorRaw` and wrap the raw bytes via a harness adapter.
pub const DataType = enum(u8) {
    BF16 = 0,
    FP32 = 1,
    FP16 = 2,
    Q8_0 = 3,
    Q4_0 = 4,
    Q6_K = 5,
    Q4_1 = 6,
    Q5_0 = 7,
    Q4_K = 8,
    Q5_K = 9,
    _,

    pub fn byteSize(self: @This(), num_elements: usize) error{Overflow}!usize {
        return switch (self) {
            .BF16, .FP16 => std.math.mul(usize, num_elements, 2),
            .FP32 => std.math.mul(usize, num_elements, 4),
            .Q8_0 => std.math.mul(usize, num_elements / Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES),
            .Q4_0 => std.math.mul(usize, num_elements / Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES),
            .Q4_1 => std.math.mul(usize, num_elements / Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES),
            .Q5_0 => std.math.mul(usize, num_elements / Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES),
            .Q5_K => std.math.mul(usize, num_elements / Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES),
            .Q6_K => std.math.mul(usize, num_elements / Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES),
            .Q4_K => std.math.mul(usize, num_elements / Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES),
            else => unreachable,
        };
    }

    pub fn numElements(self: @This(), num_bytes: usize) usize {
        return switch (self) {
            .BF16, .FP16 => num_bytes / 2,
            .FP32 => num_bytes / 4,
            .Q8_0 => (num_bytes / Q8_0_BLOCK_BYTES) * Q8_0_BLOCK_SIZE,
            .Q4_0 => (num_bytes / Q4_0_BLOCK_BYTES) * Q4_0_BLOCK_SIZE,
            .Q4_1 => (num_bytes / Q4_1_BLOCK_BYTES) * Q4_1_BLOCK_SIZE,
            .Q5_0 => (num_bytes / Q5_0_BLOCK_BYTES) * Q5_0_BLOCK_SIZE,
            .Q5_K => (num_bytes / Q5_K_BLOCK_BYTES) * Q5_K_BLOCK_SIZE,
            .Q6_K => (num_bytes / Q6_K_BLOCK_BYTES) * Q6_K_BLOCK_SIZE,
            .Q4_K => (num_bytes / Q4_K_BLOCK_BYTES) * Q4_K_BLOCK_SIZE,
            else => unreachable,
        };
    }

    pub fn toF32(self: @This(), allocator: std.mem.Allocator, data: []const u8) ![]const f32 {
        switch (self) {
            .BF16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const bf16_data: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
                const result = try allocator.alloc(f32, bf16_data.len);
                for (bf16_data, 0..) |bf16, idx| {
                    const bits: u32 = @as(u32, bf16) << 16;
                    result[idx] = @bitCast(bits);
                }
                return result;
            },
            .FP32 => {
                if (data.len % 4 != 0) return error.InvalidData;
                const f32_data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
                return try allocator.dupe(f32, f32_data);
            },
            .FP16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const f16_data: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
                const result = try allocator.alloc(f32, f16_data.len);
                for (f16_data, 0..) |f16_val, idx| {
                    result[idx] = @floatCast(f16_val);
                }
                return result;
            },
            .Q8_0 => {
                if (data.len % Q8_0_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q8_0_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q8_0_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q8_0_BLOCK_BYTES ..][0..Q8_0_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    for (0..Q8_0_BLOCK_SIZE) |elem| {
                        result[block_idx * Q8_0_BLOCK_SIZE + elem] = scale * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + elem]))));
                    }
                }
                return result;
            },
            .Q4_0 => {
                if (data.len % Q4_0_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q4_0_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q4_0_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q4_0_BLOCK_BYTES ..][0..Q4_0_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    for (0..Q4_0_BLOCK_SIZE / 2) |j| {
                        const byte = block[2 + j];
                        const low: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
                        const high: i8 = @as(i8, @intCast(byte >> 4)) - 8;
                        result[block_idx * Q4_0_BLOCK_SIZE + j] = scale * @as(f32, @floatFromInt(low));
                        result[block_idx * Q4_0_BLOCK_SIZE + j + Q4_0_BLOCK_SIZE / 2] = scale * @as(f32, @floatFromInt(high));
                    }
                }
                return result;
            },
            .Q4_1 => {
                if (data.len % Q4_1_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q4_1_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q4_1_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q4_1_BLOCK_BYTES ..][0..Q4_1_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                    for (0..Q4_1_BLOCK_SIZE / 2) |j| {
                        const byte = block[4 + j];
                        const low: f32 = @floatFromInt(@as(u8, byte & 0x0F));
                        const high: f32 = @floatFromInt(@as(u8, byte >> 4));
                        result[block_idx * Q4_1_BLOCK_SIZE + j] = low * scale + min;
                        result[block_idx * Q4_1_BLOCK_SIZE + j + Q4_1_BLOCK_SIZE / 2] = high * scale + min;
                    }
                }
                return result;
            },
            .Q5_0 => {
                if (data.len % Q5_0_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q5_0_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q5_0_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q5_0_BLOCK_BYTES ..][0..Q5_0_BLOCK_BYTES];
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const qh: u32 = std.mem.readInt(u32, block[2..6], .little);
                    for (0..Q5_0_BLOCK_SIZE / 2) |j| {
                        const byte = block[6 + j];
                        const low5: i8 = @as(i8, @intCast((byte & 0x0F) | (if ((qh >> @intCast(j)) & 1 != 0) @as(u8, 0x10) else 0))) - 16;
                        const high5: i8 = @as(i8, @intCast((byte >> 4) | (if ((qh >> @intCast(j + 16)) & 1 != 0) @as(u8, 0x10) else 0))) - 16;
                        result[block_idx * Q5_0_BLOCK_SIZE + j] = scale * @as(f32, @floatFromInt(low5));
                        result[block_idx * Q5_0_BLOCK_SIZE + j + Q5_0_BLOCK_SIZE / 2] = scale * @as(f32, @floatFromInt(high5));
                    }
                }
                return result;
            },
            .Q6_K => {
                if (data.len % Q6_K_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q6_K_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q6_K_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q6_K_BLOCK_BYTES ..][0..Q6_K_BLOCK_BYTES];
                    const ql_base = block[0..128];
                    const qh_base = block[128..192];
                    const sc_base: *const [16]i8 = @ptrCast(block[192..208]);
                    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[208..210], .little))));
                    const out = result[block_idx * Q6_K_BLOCK_SIZE ..][0..Q6_K_BLOCK_SIZE];

                    inline for (0..2) |group| {
                        const ql = ql_base[group * 64 ..];
                        const qh = qh_base[group * 32 ..];
                        const y_off = group * 128;
                        const sc_off = group * 8;

                        for (0..32) |l| {
                            const is: usize = l / 16;
                            const q1: i32 = @as(i32, (ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            const q2: i32 = @as(i32, (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                            const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                            out[y_off + l + 0] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 0])) * @as(f32, @floatFromInt(q1));
                            out[y_off + l + 32] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 2])) * @as(f32, @floatFromInt(q2));
                            out[y_off + l + 64] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 4])) * @as(f32, @floatFromInt(q3));
                            out[y_off + l + 96] = d * @as(f32, @floatFromInt(sc_base[sc_off + is + 6])) * @as(f32, @floatFromInt(q4));
                        }
                    }
                }
                return result;
            },
            .Q4_K => {
                if (data.len % Q4_K_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q4_K_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q4_K_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q4_K_BLOCK_BYTES ..][0..Q4_K_BLOCK_BYTES];
                    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                    const scales_raw = block[4..16];
                    const qs = block[16..144];
                    const out = result[block_idx * Q4_K_BLOCK_SIZE ..][0..Q4_K_BLOCK_SIZE];

                    var scales_arr: [8]u8 = undefined;
                    var mins_arr: [8]u8 = undefined;

                    scales_arr[0] = scales_raw[0] & 0x3F;
                    scales_arr[1] = scales_raw[1] & 0x3F;
                    scales_arr[2] = scales_raw[2] & 0x3F;
                    scales_arr[3] = scales_raw[3] & 0x3F;
                    mins_arr[0] = scales_raw[4] & 0x3F;
                    mins_arr[1] = scales_raw[5] & 0x3F;
                    mins_arr[2] = scales_raw[6] & 0x3F;
                    mins_arr[3] = scales_raw[7] & 0x3F;
                    scales_arr[4] = (scales_raw[8] & 0x0F) | ((scales_raw[0] >> 6) << 4);
                    scales_arr[5] = (scales_raw[9] & 0x0F) | ((scales_raw[1] >> 6) << 4);
                    scales_arr[6] = (scales_raw[10] & 0x0F) | ((scales_raw[2] >> 6) << 4);
                    scales_arr[7] = (scales_raw[11] & 0x0F) | ((scales_raw[3] >> 6) << 4);
                    mins_arr[4] = (scales_raw[8] >> 4) | ((scales_raw[4] >> 6) << 4);
                    mins_arr[5] = (scales_raw[9] >> 4) | ((scales_raw[5] >> 6) << 4);
                    mins_arr[6] = (scales_raw[10] >> 4) | ((scales_raw[6] >> 6) << 4);
                    mins_arr[7] = (scales_raw[11] >> 4) | ((scales_raw[7] >> 6) << 4);

                    for (0..4) |j| {
                        const sc1: f32 = d * @as(f32, @floatFromInt(scales_arr[j * 2]));
                        const m1: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j * 2]));
                        const sc2: f32 = d * @as(f32, @floatFromInt(scales_arr[j * 2 + 1]));
                        const m2: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j * 2 + 1]));
                        const q_base = qs[j * 32 ..];
                        const out_base = out[j * 64 ..];
                        for (0..32) |l| {
                            out_base[l] = sc1 * @as(f32, @floatFromInt(q_base[l] & 0x0F)) - m1;
                            out_base[l + 32] = sc2 * @as(f32, @floatFromInt(q_base[l] >> 4)) - m2;
                        }
                    }
                }
                return result;
            },
            .Q5_K => {
                if (data.len % Q5_K_BLOCK_BYTES != 0) return error.InvalidData;
                const num_blocks = data.len / Q5_K_BLOCK_BYTES;
                const result = try allocator.alloc(f32, num_blocks * Q5_K_BLOCK_SIZE);
                for (0..num_blocks) |block_idx| {
                    const block = data[block_idx * Q5_K_BLOCK_BYTES ..][0..Q5_K_BLOCK_BYTES];
                    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
                    const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[2..4], .little))));
                    const scales_raw = block[4..16];
                    const qh = block[16..48];
                    const qs = block[48..176];
                    const out = result[block_idx * Q5_K_BLOCK_SIZE ..][0..Q5_K_BLOCK_SIZE];

                    var scales_arr: [8]u8 = undefined;
                    var mins_arr: [8]u8 = undefined;
                    scales_arr[0] = scales_raw[0] & 0x3F;
                    scales_arr[1] = scales_raw[1] & 0x3F;
                    scales_arr[2] = scales_raw[2] & 0x3F;
                    scales_arr[3] = scales_raw[3] & 0x3F;
                    mins_arr[0] = scales_raw[4] & 0x3F;
                    mins_arr[1] = scales_raw[5] & 0x3F;
                    mins_arr[2] = scales_raw[6] & 0x3F;
                    mins_arr[3] = scales_raw[7] & 0x3F;
                    scales_arr[4] = (scales_raw[8] & 0x0F) | ((scales_raw[0] >> 6) << 4);
                    scales_arr[5] = (scales_raw[9] & 0x0F) | ((scales_raw[1] >> 6) << 4);
                    scales_arr[6] = (scales_raw[10] & 0x0F) | ((scales_raw[2] >> 6) << 4);
                    scales_arr[7] = (scales_raw[11] & 0x0F) | ((scales_raw[3] >> 6) << 4);
                    mins_arr[4] = (scales_raw[8] >> 4) | ((scales_raw[4] >> 6) << 4);
                    mins_arr[5] = (scales_raw[9] >> 4) | ((scales_raw[5] >> 6) << 4);
                    mins_arr[6] = (scales_raw[10] >> 4) | ((scales_raw[6] >> 6) << 4);
                    mins_arr[7] = (scales_raw[11] >> 4) | ((scales_raw[7] >> 6) << 4);

                    for (0..4) |j| {
                        const sc1: f32 = d * @as(f32, @floatFromInt(scales_arr[j]));
                        const m1: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j]));
                        const sc2: f32 = d * @as(f32, @floatFromInt(scales_arr[j + 4]));
                        const m2: f32 = dmin * @as(f32, @floatFromInt(mins_arr[j + 4]));
                        const q_base = qs[j * 32 ..];
                        const out_base = out[j * 64 ..];
                        for (0..32) |l| {
                            const qh_lo: u5 = @intCast((qh[l] >> @intCast(j)) & 1);
                            const qh_hi: u5 = @intCast((qh[l] >> @intCast(j + 4)) & 1);
                            const lo5 = (q_base[l] & 0x0F) | (qh_lo << 4);
                            const hi5 = (q_base[l] >> 4) | (qh_hi << 4);
                            out_base[l] = sc1 * @as(f32, @floatFromInt(lo5)) - m1;
                            out_base[l + 32] = sc2 * @as(f32, @floatFromInt(hi5)) - m2;
                        }
                    }
                }
                return result;
            },
            else => {
                log.err("unsupported data type for F32 conversion: {d}", .{@intFromEnum(self)});
                return error.UnsupportedDataType;
            },
        }
    }

    pub fn toF16(self: @This(), allocator: std.mem.Allocator, data: []const u8) ![]const f16 {
        switch (self) {
            .BF16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const bf16_data: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
                const result = try allocator.alloc(f16, bf16_data.len);
                for (bf16_data, 0..) |bf16, idx| {
                    const bits: u32 = @as(u32, bf16) << 16;
                    const f32_val: f32 = @bitCast(bits);
                    result[idx] = @floatCast(f32_val);
                }
                return result;
            },
            .FP32 => {
                if (data.len % 4 != 0) return error.InvalidData;
                const f32_data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
                const result = try allocator.alloc(f16, f32_data.len);
                for (f32_data, 0..) |f32_val, idx| {
                    result[idx] = @floatCast(f32_val);
                }
                return result;
            },
            .FP16 => {
                if (data.len % 2 != 0) return error.InvalidData;
                const f16_data: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
                return try allocator.dupe(f16, f16_data);
            },
            .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q6_K, .Q4_K, .Q5_K => {
                const f32_data = try self.toF32(allocator, data);
                defer allocator.free(f32_data);
                const result = try allocator.alloc(f16, f32_data.len);
                for (f32_data, 0..) |f32_val, idx| {
                    result[idx] = @floatCast(f32_val);
                }
                return result;
            },
            else => {
                log.err("unsupported data type for F16 conversion: {d}", .{@intFromEnum(self)});
                return error.UnsupportedDataType;
            },
        }
    }

    const Q8_0_BLOCK_SIZE = 32;
    const Q8_0_BLOCK_BYTES = 2 + Q8_0_BLOCK_SIZE;

    const Q4_0_BLOCK_SIZE = 32;
    const Q4_0_BLOCK_BYTES = 2 + Q4_0_BLOCK_SIZE / 2;

    const Q4_1_BLOCK_SIZE = 32;
    const Q4_1_BLOCK_BYTES = 2 + 2 + Q4_1_BLOCK_SIZE / 2;

    const Q5_0_BLOCK_SIZE = 32;
    const Q5_0_BLOCK_BYTES = 2 + 4 + Q5_0_BLOCK_SIZE / 2;

    const Q5_K_BLOCK_SIZE = 256;
    const Q5_K_BLOCK_BYTES = 2 + 2 + 12 + 32 + 128;

    const Q6_K_BLOCK_SIZE = 256;
    const Q6_K_BLOCK_BYTES = 128 + 64 + 16 + 2;

    const Q4_K_BLOCK_SIZE = 256;
    const Q4_K_BLOCK_BYTES = 2 + 2 + 12 + 128;
};

/// Minimal parser-native tensor view: just the raw bytes plus the dtype.
/// No conversion methods, no dependency on base.
pub const RawTensor = struct {
    data_type: DataType,
    data: []const u8,
};

/// Return the tensor's on-disk bytes plus its native `DataType` — no
/// dependency on base. Caller owns the returned buffer (free with
/// `allocator.free(raw.data)`).
pub fn getTensorRaw(self: *Self, name: []const u8) !?RawTensor {
    const allocator = self.allocator orelse {
        log.err("gguf: getTensorRaw called without allocator", .{});
        return error.IOError;
    };

    const info = self.tensor_infos.get(name) orelse return null;

    const data_type = info.dataType() catch return error.IOError;
    const byte_size = info.byteSize() catch return error.IOError;

    const abs_offset = std.math.add(u64, self.data_offset, info.offset) catch return error.IOError;
    const end_offset = std.math.add(u64, abs_offset, byte_size) catch return error.IOError;
    const file_size = self.file.getEndPos() catch return error.IOError;
    if (end_offset > file_size) return error.IOError;

    self.file.seekTo(abs_offset) catch {
        log.err("gguf: failed to seek to tensor '{s}' at offset {d}", .{ name, abs_offset });
        return error.IOError;
    };

    const data = allocator.alloc(u8, @intCast(byte_size)) catch return error.OutOfMemory;
    errdefer allocator.free(data);

    const bytes_read = self.file.readAll(data) catch {
        log.err("gguf: failed to read tensor '{s}' data", .{name});
        return error.IOError;
    };
    if (bytes_read != @as(usize, @intCast(byte_size))) {
        log.err("gguf: incomplete tensor read '{s}': got {d}, expected {d}", .{ name, bytes_read, byte_size });
        return error.IOError;
    }

    return .{
        .data_type = data_type,
        .data = data,
    };
}

// -- Public interface --

pub fn init(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    var parsed = try parseGguf(allocator, file);
    defer {
        // Free metadata - we've extracted what we need
        var it = parsed.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            freeMetadataValue(allocator, entry.value_ptr.*);
        }
        parsed.metadata.deinit(allocator);
    }

    var data_types: DataTypeSet = .none;
    {
        var ti_iter = parsed.tensor_infos.valueIterator();
        while (ti_iter.next()) |info| {
            const dtype = info.dataType() catch continue;
            switch (dtype) {
                .BF16 => data_types.BF16 = true,
                .FP32 => data_types.FP32 = true,
                .FP16 => data_types.FP16 = true,
                .Q8_0 => data_types.Q8_0 = true,
                .Q4_0 => data_types.Q4_0 = true,
                .Q4_1 => data_types.Q4_1 = true,
                .Q5_0 => data_types.Q5_0 = true,
                .Q5_K => data_types.Q5_K = true,
                .Q6_K => data_types.Q6_K = true,
                .Q4_K => data_types.Q4_K = true,
                _ => {},
            }
        }
    }

    var result = try extractConfigMap(allocator, &parsed.metadata);
    result.meta.data_types = data_types;
    errdefer {
        for (result.meta.architectures) |arch| allocator.free(arch);
        allocator.free(result.meta.architectures);
        deinitConfigMap(&result.config_map, allocator);
    }

    const vocab_result = try extractVocabulary(allocator, &parsed.metadata);
    errdefer {
        var enc_it = vocab_result.vocabulary_ptr.encoding.iterator();
        while (enc_it.next()) |entry| allocator.free(entry.key_ptr.*);
        vocab_result.vocabulary_ptr.encoding.deinit(allocator);
        var dec_it = vocab_result.vocabulary_ptr.decoding.iterator();
        while (dec_it.next()) |entry| allocator.free(entry.value_ptr.*);
        vocab_result.vocabulary_ptr.decoding.deinit(allocator);
        var merge_it = vocab_result.vocabulary_ptr.merge_index.iterator();
        while (merge_it.next()) |entry| allocator.free(entry.key_ptr.*);
        vocab_result.vocabulary_ptr.merge_index.deinit(allocator);
        allocator.destroy(vocab_result.vocabulary_ptr);
    }

    return Self{
        .meta = result.meta,
        .config_map = result.config_map,
        .vocabulary = vocab_result.vocabulary,
        .vocabulary_ptr = vocab_result.vocabulary_ptr,
        .tensor_infos = parsed.tensor_infos,
        .file = file,
        .data_offset = parsed.data_offset,
    };
}

pub fn deinit(self: Self) void {
    const allocator = self.allocator orelse return;

    // Free meta
    for (self.meta.architectures) |arch| allocator.free(arch);
    allocator.free(self.meta.architectures);

    // Free config_map
    var map = self.config_map;
    deinitConfigMap(&map, allocator);

    // Free vocabulary
    var enc_it = self.vocabulary_ptr.encoding.iterator();
    while (enc_it.next()) |entry| allocator.free(entry.key_ptr.*);
    self.vocabulary_ptr.encoding.deinit(allocator);

    var dec_it = self.vocabulary_ptr.decoding.iterator();
    while (dec_it.next()) |entry| allocator.free(entry.value_ptr.*);
    self.vocabulary_ptr.decoding.deinit(allocator);

    var merge_it = self.vocabulary_ptr.merge_index.iterator();
    while (merge_it.next()) |entry| allocator.free(entry.key_ptr.*);
    self.vocabulary_ptr.merge_index.deinit(allocator);

    if (self.vocabulary_ptr.post_processor) |pp| {
        switch (pp) {
            .template => |template| allocator.free(template),
            .sequence => |seq| allocator.free(seq),
        }
    }

    allocator.destroy(self.vocabulary_ptr);

    // Free tensor info keys
    var ti_it = @constCast(&self.tensor_infos).iterator();
    while (ti_it.next()) |entry| allocator.free(entry.key_ptr.*);
    @constCast(&self.tensor_infos).deinit(allocator);
}

test {
    const file = try std.fs.cwd().openFile("test_models/TinyStories-656K-GGUF/TinyStories-656K.f16.gguf", .{});
    defer file.close();

    var data = try init(testing.allocator, file);
    data.allocator = testing.allocator;
    defer data.deinit();

    // Meta
    try testing.expectEqual(1, data.meta.architectures.len);
    try testing.expectEqualStrings("llama", data.meta.architectures[0]);

    // ConfigMap: arch-prefixed keys should be stripped
    try testing.expectEqual(2, configGetUint(data.config_map, "block_count").?);
    try testing.expectEqual(128, configGetUint(data.config_map, "embedding_length").?);
    try testing.expectEqual(8, configGetUint(data.config_map, "attention.head_count").?);
    try testing.expectEqual(4, configGetUint(data.config_map, "attention.head_count_kv").?);
    try testing.expectEqual(384, configGetUint(data.config_map, "feed_forward_length").?);
    try testing.expectEqual(2048, data.meta.vocabulary_size);

    // Vocabulary
    try testing.expectEqual(26, data.vocabulary.encoding.get("A").?);
    try testing.expectEqualStrings("A", data.vocabulary.decoding.get(26).?);

    // Tensor reading (direct GGUF name)
    const embeddings = try data.getTensorRaw("token_embd.weight");
    defer testing.allocator.free(embeddings.?.data);

    const embeddings_f32 = try embeddings.?.data_type.toF32(testing.allocator, embeddings.?.data);
    defer testing.allocator.free(embeddings_f32);

    try testing.expectEqual(embeddings.?.data.len, embeddings_f32.len * 2);
}

test "Q8_0 GGUF loading" {
    const file = try std.fs.cwd().openFile("test_models/TinyStories-656K-Q8_0-GGUF/tinystories-656k-q8_0.gguf", .{});
    defer file.close();

    var data = try init(testing.allocator, file);
    data.allocator = testing.allocator;
    defer data.deinit();

    try testing.expectEqual(2, configGetUint(data.config_map, "block_count").?);
    try testing.expectEqual(128, configGetUint(data.config_map, "embedding_length").?);

    // Read a Q8_0 weight tensor
    const q_proj = try data.getTensorRaw("blk.0.attn_q.weight");
    defer testing.allocator.free(q_proj.?.data);

    try testing.expect(q_proj != null);
    try testing.expectEqual(DataType.Q8_0, q_proj.?.data_type);

    // Verify dequantization to F32
    const f32_data = try q_proj.?.data_type.toF32(testing.allocator, q_proj.?.data);
    defer testing.allocator.free(f32_data);
    // 128 * 128 = 16384 elements
    try testing.expectEqual(16384, f32_data.len);
}

// -- Config types (used by config_map and meta fields) --

pub const ConfigValue = union(enum) {
    uint: usize,
    float: f32,
    boolean: bool,
    string: []const u8,
    uint_array: []const usize,
    string_array: []const []const u8,
};

pub const ConfigMap = std.StringHashMapUnmanaged(ConfigValue);

pub const Meta = struct {
    architectures: []const []const u8,
    data_types: DataTypeSet,
    vocabulary_size: usize,
    max_len: usize,
    bos_token_id: usize,
    eos_token_id: usize,
};

pub const DataTypeSet = struct {
    BF16: bool = false,
    FP32: bool = false,
    FP16: bool = false,
    Q8_0: bool = false,
    Q4_0: bool = false,
    Q4_1: bool = false,
    Q5_0: bool = false,
    Q5_K: bool = false,
    Q6_K: bool = false,
    Q4_K: bool = false,

    pub const none: DataTypeSet = .{};
};

pub fn configGetUint(map: ConfigMap, key: []const u8) ?usize {
    const val = map.get(key) orelse return null;
    return switch (val) {
        .uint => |value| value,
        else => null,
    };
}

pub fn deinitConfigMap(map: *ConfigMap, allocator: std.mem.Allocator) void {
    var it = map.iterator();
    while (it.next()) |entry| {
        switch (entry.value_ptr.*) {
            .string => |str| allocator.free(str),
            .uint_array => |arr| allocator.free(arr),
            .string_array => |arr| {
                for (arr) |str| allocator.free(str);
                allocator.free(arr);
            },
            else => {},
        }
        allocator.free(entry.key_ptr.*);
    }
    map.deinit(allocator);
}

const Vocabulary = @import("base").Vocabulary;

const log = std.log.scoped(.infer);

const std = @import("std");
const testing = std.testing;

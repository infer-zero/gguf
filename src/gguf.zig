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

    fn dataType(self: TensorInfo) !Tensor.DataType {
        return switch (self.gguf_type) {
            0 => .FP32,
            1 => .FP16,
            2 => .Q4_0,
            3 => .Q4_1,
            8 => .Q8_0,
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

    for (tokens, 0..) |token_val, idx| {
        const token_str = token_val.getString() orelse continue;
        const token_id: Vocabulary.TokenID = @intCast(idx);

        const enc_key = try allocator.dupe(u8, token_str);
        errdefer allocator.free(enc_key);
        const dec_val = try allocator.dupe(u8, token_str);
        errdefer allocator.free(dec_val);

        try encoding.put(allocator, enc_key, token_id);
        try decoding.put(allocator, token_id, dec_val);

        // Check if this is the unknown token (type == 2)
        if (token_types) |types| {
            if (idx < types.len) {
                if (types[idx].getUint()) |token_type| {
                    if (token_type == 2) {
                        unknown_token = dec_val;
                    }
                }
            }
        }
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
    };

    return .{
        .vocabulary_ptr = vocab_ptr,
        .vocabulary = vocab_ptr.*,
    };
}

// -- Public tensor reading --

/// Read a tensor by name, returning its raw data and type. Returns null if not found.
pub fn getTensor(self: *Self, name: []const u8) !?Tensor {
    const allocator = self.allocator orelse {
        log.err("gguf: getTensor called without allocator", .{});
        return error.IOError;
    };

    const info = self.tensor_infos.get(name) orelse return null;

    const data_type = info.dataType() catch return error.IOError;
    const byte_size = info.byteSize() catch return error.IOError;

    // Seek to the tensor data (checked arithmetic to prevent overflow from malicious offsets)
    const abs_offset = std.math.add(u64, self.data_offset, info.offset) catch return error.IOError;
    const end_offset = std.math.add(u64, abs_offset, byte_size) catch return error.IOError;
    const file_size = self.file.getEndPos() catch return error.IOError;
    if (end_offset > file_size) return error.IOError;

    self.file.seekTo(abs_offset) catch {
        log.err("gguf: failed to seek to tensor '{s}' at offset {d}", .{ name, abs_offset });
        return error.IOError;
    };

    // Read the data
    const data = allocator.alloc(u8, @intCast(byte_size)) catch return error.OutOfMemory;
    errdefer allocator.free(data);

    const bytes_read = self.file.readAll(data) catch {
        log.err("gguf: failed to read tensor '{s}' data", .{name});
        return error.IOError;
    };
    if (bytes_read != @as(usize, @intCast(byte_size))) {
        log.err("gguf: tensor '{s}' read incomplete: got {d}/{d} bytes", .{ name, bytes_read, byte_size });
        return error.IOError;
    }

    return Tensor{
        .data_type = data_type,
        .data = data,
    };
}

/// Free a tensor previously returned by getTensor.
pub fn releaseTensor(self: *Self, tensor: ?Tensor) void {
    if (tensor) |tens| {
        if (self.allocator) |allocator| {
            tens.deinit(allocator);
        }
    }
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
                .Q6_K => data_types.Q6_K = true,
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
    const embeddings = try data.getTensor("token_embd.weight");
    defer data.releaseTensor(embeddings);

    const embeddings_f32 = try embeddings.?.toF32(testing.allocator);
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
    const q_proj = try data.getTensor("blk.0.attn_q.weight");
    defer data.releaseTensor(q_proj);

    try testing.expect(q_proj != null);
    try testing.expectEqual(Tensor.DataType.Q8_0, q_proj.?.data_type);

    // Verify dequantization to F32
    const f32_data = try q_proj.?.toF32(testing.allocator);
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
    Q6_K: bool = false,

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
const Tensor = @import("base").Tensor;

const log = std.log.scoped(.infer);

const std = @import("std");
const testing = std.testing;

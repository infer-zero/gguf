//! Tokenizer data extracted straight from GGUF metadata — no derived
//! maps or synthesized processors. The parser owns the raw shape;
//! `harness/src/adapters.zig::vocabularyOwned` builds the lookup
//! structures the runtime tokenizer needs.
//!
//! All string slices are owned by the allocator passed to `extract`;
//! `deinit` frees them.

tokens: []const []const u8 = &.{},
token_types: ?[]const TokenType = null,
merges: []const Merge = &.{},
model: []const u8 = "",
add_bos_token: bool = true,
bos_token_id: TokenID = 1,
unknown_token_id: ?TokenID = null,

pub const TokenID = u32;

/// Per-token classification from `tokenizer.ggml.token_type`. The numeric
/// values match the GGUF on-disk encoding.
pub const TokenType = enum(u8) {
    normal = 1,
    unknown = 2,
    /// Model-intrinsic markers like `<s>`, `<|im_start|>`. Must be matched
    /// literally before BPE, otherwise chat-template boundaries get split.
    control = 3,
    /// User-added tokens (rare in GGUF). Also literal-match, never BPE'd.
    user_defined = 4,
    unused = 5,
    /// SentencePiece byte-fallback tokens like `<0x41>`.
    byte = 6,
    _,
};

pub const Merge = struct {
    first: []const u8,
    second: []const u8,
};

pub fn init(allocator: std.mem.Allocator, metadata: Metadata) !@This() {
    const tokens_val = metadata.get("tokenizer.ggml.tokens") orelse {
        log.err("gguf: missing 'tokenizer.ggml.tokens' metadata", .{});
        return error.ReaderError;
    };
    if (tokens_val != .array) {
        log.err("gguf: 'tokenizer.ggml.tokens' is not an array", .{});
        return error.ReaderError;
    }
    const token_vals = tokens_val.array.values;

    var tokens_buf = try allocator.alloc([]const u8, token_vals.len);
    var tokens_loaded: usize = 0;
    errdefer {
        for (tokens_buf[0..tokens_loaded]) |s| allocator.free(s);
        allocator.free(tokens_buf);
    }
    for (token_vals, 0..) |val, idx| {
        const s = val.getString() orelse "";
        tokens_buf[idx] = try allocator.dupe(u8, s);
        tokens_loaded = idx + 1;
    }

    const token_types: ?[]const TokenType = blk: {
        const val = metadata.get("tokenizer.ggml.token_type") orelse break :blk null;
        if (val != .array) break :blk null;
        const src = val.array.values;
        const buf = try allocator.alloc(TokenType, src.len);
        for (src, 0..) |elem, idx| {
            const raw = elem.getUint() orelse 0;
            buf[idx] = @enumFromInt(@as(u8, @truncate(raw)));
        }
        break :blk buf;
    };
    errdefer if (token_types) |t| allocator.free(t);

    const merges = try readMerges(allocator, metadata);
    errdefer {
        for (merges) |m| {
            allocator.free(m.first);
            allocator.free(m.second);
        }
        allocator.free(merges);
    }

    const model = blk: {
        const val = metadata.get("tokenizer.ggml.model") orelse break :blk try allocator.dupe(u8, "");
        const s = val.getString() orelse "";
        break :blk try allocator.dupe(u8, s);
    };
    errdefer allocator.free(model);

    const add_bos_token = blk: {
        if (metadata.get("tokenizer.ggml.add_bos_token")) |val| {
            break :blk val == .bool_val and val.bool_val;
        }
        break :blk true; // GGUF default: add BOS
    };

    const bos_token_id: TokenID = @intCast(metadataUsize(metadata, "tokenizer.ggml.bos_token_id") orelse 1);
    const unknown_token_id: ?TokenID = if (metadataUsize(metadata, "tokenizer.ggml.unknown_token_id")) |id|
        @intCast(id)
    else
        null;

    return .{
        .tokens = tokens_buf,
        .token_types = token_types,
        .merges = merges,
        .model = model,
        .add_bos_token = add_bos_token,
        .bos_token_id = bos_token_id,
        .unknown_token_id = unknown_token_id,
    };
}

pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
    for (self.tokens) |s| allocator.free(s);
    allocator.free(self.tokens);
    if (self.token_types) |t| allocator.free(t);
    for (self.merges) |m| {
        allocator.free(m.first);
        allocator.free(m.second);
    }
    allocator.free(self.merges);
    allocator.free(self.model);
}

fn readMerges(allocator: std.mem.Allocator, metadata: Metadata) ![]const Merge {
    const val = metadata.get("tokenizer.ggml.merges") orelse return &.{};
    if (val != .array) return &.{};
    const src = val.array.values;

    var buf = try allocator.alloc(Merge, src.len);
    var loaded: usize = 0;
    errdefer {
        for (buf[0..loaded]) |m| {
            allocator.free(m.first);
            allocator.free(m.second);
        }
        allocator.free(buf);
    }

    var out: usize = 0;
    for (src) |elem| {
        const merge_str = elem.getString() orelse continue;
        const split = std.mem.indexOfScalar(u8, merge_str, ' ') orelse continue;
        const first = try allocator.dupe(u8, merge_str[0..split]);
        errdefer allocator.free(first);
        const second = try allocator.dupe(u8, merge_str[split + 1 ..]);
        buf[out] = .{ .first = first, .second = second };
        out += 1;
        loaded = out;
    }

    if (out != src.len) {
        return try allocator.realloc(buf, out);
    }
    return buf;
}

fn metadataUsize(metadata: Metadata, key: []const u8) ?usize {
    if (metadata.get(key)) |val| {
        if (val.getUint()) |uint_val| return @intCast(uint_val);
    }
    return null;
}

const log = std.log.scoped(.infer);
const std = @import("std");
const Metadata = @import("metadata.zig");

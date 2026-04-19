//! Headline facts every consumer reads, pulled directly from the
//! parsed `Metadata`. `init` looks up `general.architecture` and
//! `tokenizer.ggml.*` globally; architecture-specific keys like
//! `context_length` are looked up under the arch prefix
//! (e.g. `llama.context_length`).

architectures: []const []const u8,
vocabulary_size: usize,
max_len: usize,
bos_token_id: usize,
eos_token_id: usize,

pub fn init(allocator: std.mem.Allocator, metadata: Metadata) !@This() {
    const arch = blk: {
        if (metadata.get("general.architecture")) |val| {
            if (val.getString()) |s| break :blk s;
        }
        log.err("gguf: missing 'general.architecture' metadata", .{});
        return error.ReaderError;
    };

    const architectures = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(architectures);
    architectures[0] = try allocator.dupe(u8, arch);
    errdefer allocator.free(architectures[0]);

    // Prefer the tokens-array length (authoritative); fall back to
    // `{arch}.vocab_size` which some older GGUFs set explicitly.
    const vocabulary_size: usize = blk: {
        if (metadata.get("tokenizer.ggml.tokens")) |val| {
            if (val == .array) break :blk val.array.values.len;
        }
        break :blk archUint(metadata, arch, "vocab_size") orelse 0;
    };

    return .{
        .architectures = architectures,
        .vocabulary_size = vocabulary_size,
        .max_len = archUint(metadata, arch, "context_length") orelse 2048,
        .bos_token_id = metadataUint(metadata, "tokenizer.ggml.bos_token_id") orelse 1,
        .eos_token_id = metadataUint(metadata, "tokenizer.ggml.eos_token_id") orelse 2,
    };
}

pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
    for (self.architectures) |a| allocator.free(a);
    allocator.free(self.architectures);
}

fn metadataUint(metadata: Metadata, key: []const u8) ?usize {
    const val = metadata.get(key) orelse return null;
    const uint_val = val.getUint() orelse return null;
    return @intCast(uint_val);
}

/// Look up an architecture-prefixed uint, e.g. `llama.context_length`.
/// Composes the full key on the stack so there's no heap traffic.
fn archUint(metadata: Metadata, arch: []const u8, short_key: []const u8) ?usize {
    var buf: [256]u8 = undefined;
    const total = arch.len + 1 + short_key.len;
    if (total > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    buf[arch.len] = '.';
    @memcpy(buf[arch.len + 1 ..][0..short_key.len], short_key);
    return metadataUint(metadata, buf[0..total]);
}

const log = std.log.scoped(.infer);
const std = @import("std");
const Metadata = @import("metadata.zig");

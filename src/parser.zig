//! A parsed GGUF file. `init` walks the file top-to-bottom into five
//! sections — `header`, `metadata`, `config`, `tokenizer`, `tensors` —
//! and holds them until `deinit`. The underlying file handle is
//! borrowed; the caller owns its lifecycle.

const Self = @This();

allocator: std.mem.Allocator,
header: Header,
metadata: Metadata,
config: Config,
tokenizer: Tokenizer,
tensors: Tensors,

pub fn init(io: std.Io, allocator: std.mem.Allocator, file: std.Io.File) !Self {
    var buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    const reader = &file_reader.interface;

    const header = try Header.init(reader);

    var metadata = try Metadata.init(allocator, reader, header.metadata_kv_count);
    errdefer metadata.deinit();

    const default_alignment = 32;
    const alignment: u64 = blk: {
        if (metadata.get("general.alignment")) |val| {
            if (val.getUint()) |align_value| break :blk align_value;
        }
        break :blk default_alignment;
    };

    var tensors = try Tensors.init(io, allocator, &file_reader, alignment, header.tensor_count);
    errdefer tensors.deinit();

    var config = try Config.init(allocator, metadata);
    errdefer config.deinit(allocator);

    var tok = try Tokenizer.init(allocator, metadata);
    errdefer tok.deinit(allocator);

    return .{
        .allocator = allocator,
        .header = header,
        .metadata = metadata,
        .config = config,
        .tokenizer = tok,
        .tensors = tensors,
    };
}

pub fn deinit(self: *Self) void {
    self.config.deinit(self.allocator);
    self.tokenizer.deinit(self.allocator);
    self.tensors.deinit();
    self.metadata.deinit();
}

/// Read tensor `name`'s raw bytes and element type. Shortcut for
/// `self.tensors.get(name)`.
pub fn getTensorRaw(self: *Self, name: []const u8) !?Tensors.Raw {
    return self.tensors.get(name);
}

test "F16 GGUF loading" {
    const io = testing.io;
    const file = try std.Io.Dir.cwd().openFile(io, "test_models/TinyStories-656K-GGUF/TinyStories-656K.f16.gguf", .{});
    defer file.close(io);

    var parser = try init(io, testing.allocator, file);
    defer parser.deinit();

    try testing.expectEqual(1, parser.config.architectures.len);
    try testing.expectEqualStrings("llama", parser.config.architectures[0]);

    try testing.expectEqual(2, parser.metadata.get("llama.block_count").?.getUint().?);
    try testing.expectEqual(128, parser.metadata.get("llama.embedding_length").?.getUint().?);
    try testing.expectEqual(8, parser.metadata.get("llama.attention.head_count").?.getUint().?);
    try testing.expectEqual(4, parser.metadata.get("llama.attention.head_count_kv").?.getUint().?);
    try testing.expectEqual(384, parser.metadata.get("llama.feed_forward_length").?.getUint().?);
    try testing.expectEqual(2048, parser.config.vocabulary_size);

    try testing.expectEqualStrings("A", parser.tokenizer.tokens[26]);

    const embeddings = try parser.getTensorRaw("token_embd.weight");
    defer testing.allocator.free(embeddings.?.data);
}

test "Q8_0 GGUF loading" {
    const io = testing.io;
    const file = try std.Io.Dir.cwd().openFile(io, "test_models/TinyStories-656K-Q8_0-GGUF/tinystories-656k-q8_0.gguf", .{});
    defer file.close(io);

    var parser = try init(io, testing.allocator, file);
    defer parser.deinit();

    try testing.expectEqual(2, parser.metadata.get("llama.block_count").?.getUint().?);
    try testing.expectEqual(128, parser.metadata.get("llama.embedding_length").?.getUint().?);

    const q_proj = try parser.getTensorRaw("blk.0.attn_q.weight");
    defer testing.allocator.free(q_proj.?.data);

    try testing.expect(q_proj != null);
    try testing.expectEqual(Tensors.DataType.Q8_0, q_proj.?.data_type);
}

const std = @import("std");
const testing = std.testing;

const Header = @import("header.zig");
const Metadata = @import("metadata.zig");
const Config = @import("config.zig");
const Tokenizer = @import("tokenizer.zig");
const Tensors = @import("tensors.zig");

//! The GGUF v3 fixed header — the four values at the start of every
//! file. No allocations, so there's no `deinit`.

magic: u32,
version: u32,
tensor_count: u64,
metadata_kv_count: u64,

/// Read and validate the header. Fails if the magic doesn't match
/// `"GGUF"` or the version isn't 3.
pub fn init(reader: *std.Io.Reader) !@This() {
    const magic = try reader.takeInt(u32, .little);
    if (magic != gguf_magic) {
        log.err("gguf: invalid magic: 0x{x} (expected 0x{x})", .{ magic, gguf_magic });
        return error.ReaderError;
    }

    const version = try reader.takeInt(u32, .little);
    if (version != supported_version) {
        log.err("gguf: unsupported version: {d} (expected {d})", .{ version, supported_version });
        return error.ReaderError;
    }

    return .{
        .magic = magic,
        .version = version,
        .tensor_count = try reader.takeInt(u64, .little),
        .metadata_kv_count = try reader.takeInt(u64, .little),
    };
}

const gguf_magic = 0x46554747;
const supported_version = 3;

const log = std.log.scoped(.infer);
const std = @import("std");

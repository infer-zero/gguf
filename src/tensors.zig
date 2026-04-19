//! The GGUF tensor-info block plus the tensor-data section it points
//! at. `init` reads every tensor-info entry into `map` and captures the
//! aligned `data_offset`. `get(name)` seeks into the underlying file
//! and returns the tensor's raw bytes; `deinit` frees the info map.
//! The file handle is borrowed — the caller is responsible for closing
//! it.

allocator: std.mem.Allocator,
map: Map,
file: std.fs.File,
data_offset: u64,

pub const Map = std.StringHashMapUnmanaged(Info);

/// One tensor-info entry from the GGUF tensor table: shape, element
/// type (`gguf_type`), and byte offset into the tensor-data section.
pub const Info = struct {
    n_dimensions: u32,
    dimensions: [4]u64,
    gguf_type: u32,
    offset: u64,

    pub fn dataType(self: Info) !DataType {
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

    pub fn byteSize(self: Info) !u64 {
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

/// A tensor's on-disk bytes plus its element type. Dequant via
/// `base.Tensor.toF32`/`toF16` after wrapping with
/// `harness/src/adapters.zig::rawToTensor`.
pub const Raw = struct {
    data_type: DataType,
    data: []const u8,
};

/// Tensor element type. Numeric values mirror `base.Tensor.DataType`
/// so `@enumFromInt(@intFromEnum(x))` converts between the two — see
/// `harness/src/adapters.zig::rawToTensor`, which is how consumers
/// reach a dequantizable `base.Tensor`.
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

/// Read the tensor-info block from `file_reader` and compute the
/// aligned start of the tensor-data section using `alignment` and the
/// reader's current logical position.
pub fn init(
    allocator: std.mem.Allocator,
    file_reader: *std.fs.File.Reader,
    alignment: u64,
    tensor_count: u64,
) !@This() {
    var self: @This() = .{
        .allocator = allocator,
        .map = .empty,
        .file = file_reader.file,
        .data_offset = 0,
    };
    errdefer self.deinit();

    const reader = &file_reader.interface;

    for (0..@intCast(tensor_count)) |_| {
        const name = try readString(allocator, reader);
        errdefer allocator.free(name);

        const n_dimensions = try reader.takeInt(u32, .little);
        if (n_dimensions > 4) {
            log.err("gguf: tensor has {d} dimensions (max 4)", .{n_dimensions});
            return error.ReaderError;
        }

        var dimensions: [4]u64 = .{ 0, 0, 0, 0 };
        for (0..n_dimensions) |dim_index| {
            dimensions[dim_index] = try reader.takeInt(u64, .little);
        }

        const gguf_type = try reader.takeInt(u32, .little);
        const offset = try reader.takeInt(u64, .little);

        try self.map.put(allocator, name, .{
            .n_dimensions = n_dimensions,
            .dimensions = dimensions,
            .gguf_type = gguf_type,
            .offset = offset,
        });
    }

    const current_pos = file_reader.logicalPos();
    self.data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

    return self;
}

pub fn deinit(self: *@This()) void {
    var it = self.map.iterator();
    while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
    self.map.deinit(self.allocator);
}

/// Read tensor `name`'s raw bytes and element type. Caller owns the
/// returned buffer — free with `allocator.free(raw.data)`. Returns
/// `null` if the tensor is not present.
pub fn get(self: *@This(), name: []const u8) !?Raw {
    const info = self.map.get(name) orelse return null;

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

    const data = self.allocator.alloc(u8, @intCast(byte_size)) catch return error.OutOfMemory;
    errdefer self.allocator.free(data);

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

/// Read a length-prefixed GGUF string (tensor name). Sanity-caps at
/// 1 MiB so a corrupt file can't trigger a giant allocation.
fn readString(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]const u8 {
    const len = try reader.takeInt(u64, .little);
    if (len > 1024 * 1024) {
        log.err("gguf: string length {d} exceeds sanity limit", .{len});
        return error.ReaderError;
    }
    return try reader.readAlloc(allocator, @intCast(len));
}

const log = std.log.scoped(.infer);
const std = @import("std");

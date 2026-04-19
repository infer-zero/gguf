//! The GGUF metadata KV block — parsed and owned. `init` reads
//! `kv_count` entries from a `std.Io.Reader`; `deinit` frees every
//! owned string and nested array. Consumers look up raw values via
//! `get`, or pass the whole `Metadata` to the other sections'
//! constructors.

allocator: std.mem.Allocator,
map: Map,

pub const Map = std.StringHashMapUnmanaged(Value);

/// A single GGUF metadata value: one of the 13 on-disk KV types
/// (scalars, bool, string, or a homogeneous array). Owned strings and
/// array buffers live under the parent `Metadata.allocator`.
pub const Value = union(enum) {
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
    array: Array,

    pub const Array = struct {
        elem_type: u32,
        values: []const Value,
    };

    pub fn getUint(self: Value) ?u64 {
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

    pub fn getFloat(self: Value) ?f64 {
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

    pub fn getString(self: Value) ?[]const u8 {
        return switch (self) {
            .string => |value| value,
            else => null,
        };
    }
};

pub fn init(allocator: std.mem.Allocator, reader: *std.Io.Reader, kv_count: u64) !@This() {
    var self: @This() = .{
        .allocator = allocator,
        .map = .empty,
    };
    errdefer self.deinit();

    for (0..@intCast(kv_count)) |_| {
        const key = try readString(allocator, reader);
        errdefer allocator.free(key);
        const value_type = try reader.takeInt(u32, .little);
        const value = try readValue(allocator, reader, value_type, 0);
        errdefer freeValue(allocator, value);
        try self.map.put(allocator, key, value);
    }

    return self;
}

pub fn deinit(self: *@This()) void {
    var it = self.map.iterator();
    while (it.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        freeValue(self.allocator, entry.value_ptr.*);
    }
    self.map.deinit(self.allocator);
}

/// Look up a GGUF metadata key. Keys are the full on-disk names —
/// `general.architecture`, `llama.block_count`, `tokenizer.ggml.tokens`,
/// etc. (no prefix stripping; that's `ConfigMap`'s job).
pub fn get(self: @This(), key: []const u8) ?Value {
    return self.map.get(key);
}

/// Read a length-prefixed GGUF string. Sanity-caps at 1 MiB so a
/// corrupt file can't trigger a giant allocation.
fn readString(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]const u8 {
    const len = try reader.takeInt(u64, .little);
    if (len > 1024 * 1024) {
        log.err("gguf: string length {d} exceeds sanity limit", .{len});
        return error.ReaderError;
    }
    return try reader.readAlloc(allocator, @intCast(len));
}

/// Read one value of GGUF type `value_type`. Recurses into array
/// elements; `depth` guards against pathological nesting.
fn readValue(allocator: std.mem.Allocator, reader: *std.Io.Reader, value_type: u32, depth: u32) !Value {
    if (depth > 4) {
        log.err("gguf: metadata recursion depth exceeds limit", .{});
        return error.ReaderError;
    }
    return switch (value_type) {
        0 => .{ .uint8 = try reader.takeByte() }, // UINT8
        1 => .{ .int8 = try reader.takeByteSigned() }, // INT8
        2 => .{ .uint16 = try reader.takeInt(u16, .little) }, // UINT16
        3 => .{ .int16 = try reader.takeInt(i16, .little) }, // INT16
        4 => .{ .uint32 = try reader.takeInt(u32, .little) }, // UINT32
        5 => .{ .int32 = try reader.takeInt(i32, .little) }, // INT32
        6 => .{ .float32 = @bitCast(try reader.takeInt(u32, .little)) }, // FLOAT32
        7 => .{ .bool_val = (try reader.takeByte()) != 0 }, // BOOL
        8 => .{ .string = try readString(allocator, reader) }, // STRING
        9 => blk: { // ARRAY
            const elem_type = try reader.takeInt(u32, .little);
            const count = try reader.takeInt(u64, .little);
            if (count > 10 * 1024 * 1024) {
                log.err("gguf: array count {d} exceeds sanity limit", .{count});
                return error.ReaderError;
            }
            const values = try allocator.alloc(Value, @intCast(count));
            errdefer allocator.free(values);
            for (0..@intCast(count)) |index| {
                values[index] = try readValue(allocator, reader, elem_type, depth + 1);
            }
            break :blk .{ .array = .{ .elem_type = elem_type, .values = values } };
        },
        10 => .{ .uint64 = try reader.takeInt(u64, .little) }, // UINT64
        11 => .{ .int64 = try reader.takeInt(i64, .little) }, // INT64
        12 => .{ .float64 = @bitCast(try reader.takeInt(u64, .little)) }, // FLOAT64
        else => {
            log.err("gguf: unsupported metadata value type: {d}", .{value_type});
            return error.ReaderError;
        },
    };
}

fn freeValue(allocator: std.mem.Allocator, value: Value) void {
    switch (value) {
        .string => |str| allocator.free(str),
        .array => |arr| {
            for (arr.values) |element| freeValue(allocator, element);
            allocator.free(arr.values);
        },
        else => {},
    }
}

const log = std.log.scoped(.infer);
const std = @import("std");

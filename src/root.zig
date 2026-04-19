//! Public surface of the gguf package. `Parser` owns the lifecycle and
//! all parsed sections (see `parser.zig`); everything else here is a
//! type re-export for consumer convenience.

pub const Parser = @import("parser.zig");

pub const Header = @import("header.zig");
pub const Metadata = @import("metadata.zig");
pub const Config = @import("config.zig");
pub const Tokenizer = @import("tokenizer.zig");
pub const Tensors = @import("tensors.zig");

test {
    _ = Parser;
}

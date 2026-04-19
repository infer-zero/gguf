# gguf

A standalone Zig parser for the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) v3 model format used by llama.cpp and the wider GGML ecosystem. No runtime dependencies beyond Zig's std library.

## What it covers

| Section | What it exposes |
|---|---|
| `Header` | Magic, version, tensor count, metadata KV count. |
| `Metadata` | Every raw KV pair with 13-type `Value` union (`uint8`/`int8`/…/`string`/`array`) and `getUint`/`getFloat`/`getString` coercion helpers. |
| `Config` | Headline fields extracted from metadata: `architectures`, `vocabulary_size`, `max_len`, `bos_token_id`, `eos_token_id`. |
| `Tokenizer` | Raw on-disk shape: `tokens` array, optional `token_types`, BPE `merges`, `model` name (`gpt2` / `llama` / …), `add_bos_token`, `bos_token_id`, `unknown_token_id`. |
| `Tensors` | Tensor-info map + file handle + aligned data offset; `get(name)` reads raw bytes + `DataType`. |

Supported tensor types: `BF16`, `FP32`, `FP16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`. Reads are streamed through a buffered `std.Io.Reader`.

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/gguf
```

In `build.zig`:

```zig
const gguf_dep = b.dependency("gguf", .{ .target = target, .optimize = optimize });
my_mod.addImport("gguf", gguf_dep.module("gguf"));
```

In code — `Parser` owns the whole parsed file. The caller owns the `std.fs.File` handle (the parser borrows it for tensor reads).

```zig
const gguf = @import("gguf");

const file = try std.fs.cwd().openFile("model.gguf", .{});
defer file.close();

var parser = try gguf.Parser.init(allocator, file);
defer parser.deinit();

// Headline facts
const arch = parser.config.architectures[0];
const vocab = parser.config.vocabulary_size;

// Any raw key
const block_count = parser.metadata.get("llama.block_count").?.getUint().?;

// Tensor bytes
const raw = try parser.getTensorRaw("token_embd.weight");
defer allocator.free(raw.?.data);
// raw.?.data_type is a Tensors.DataType (BF16 / Q8_0 / …)
```

## Public surface

```zig
gguf.Parser           // lifecycle: init(allocator, file) / deinit / getTensorRaw
gguf.Header           // fixed header
gguf.Metadata         // raw KV block
gguf.Config           // extracted headline fields
gguf.Tokenizer        // raw tokenizer data
gguf.Tensors          // tensor info + data reader
gguf.Tensors.Info     // one tensor's shape + type + offset
gguf.Tensors.Raw      // { data_type, data }
gguf.Tensors.DataType // element type enum
```

See `src/root.zig` for the full re-export list.

## Testing

```bash
zig build test
```

Two tests exercise the full parse path against the TinyStories F16 and Q8_0 fixtures under `test_models/`.

## License

MIT

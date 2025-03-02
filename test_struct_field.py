import time

import polars as pl

n = 100

# df = pl.DataFrame({"a": [[{"a": 1, "b": 2}, {"a": 3, "b": 4}]] * 1_000_000})
df = pl.read_parquet("/Users/borchero/git/aramis/df.parquet")
df = pl.concat([df] * 10)

print(
    df.lazy()
    .with_columns(b=pl.col("a").list.eval(pl.element().struct.field("a")))
    .collect_schema()
)

tic = time.time()
for _ in range(n):
    df.with_columns(b=pl.col("a").list.eval(pl.element().struct.field("a")))
toc = time.time()
print(f"`.list.eval` takes: {(toc - tic) / n}")

print(df.lazy().with_columns(b=pl.col("a").list.struct_field("a")).collect_schema())

tic = time.time()
for _ in range(n):
    df.with_columns(b=pl.col("a").list.struct_field("a"))
toc = time.time()
print(f"`.list.struct_field` takes: {(toc - tic) / n}")

import datetime as dt

import polars as pl

df = pl.DataFrame({"a": [[1, 2, 3]], "n": [2]})
print(df)
out = df.select(pl.col("a").repeat_by(pl.col("n")))
print(out)

df = pl.DataFrame({"a": [["a", "b", "c"]], "n": [2]})
print(df)
out = df.select(pl.col("a").repeat_by(pl.col("n")))
print(out)

df = pl.DataFrame(
    {"a": [["a", "b", None]], "n": [2]},
    schema={"a": pl.List(pl.Enum(["a", "b"])), "n": pl.UInt32},
)
print(df)
out = df.select(pl.col("a").repeat_by(pl.col("n")))
print(out)

# df = pl.DataFrame({"a": [{"a": 5, "b": [1, 2, 3]}], "n": [2]})
# print(df)
# out = df.select(pl.col("a").repeat_by(pl.col("n")))
# print(out)

df = pl.DataFrame({"a": [{"a": [1, 2, 3], "b": [4, 5]}], "n": [2]})
print(df)
out = df.select(pl.col("a").repeat_by(pl.col("n")))
print(out)


df = pl.DataFrame(
    {
        "icds": [
            [
                {"icd": "A123", "location": "L", "date": dt.date(2020, 1, 1)},
                {"icd": "B456", "location": None, "date": dt.date(2020, 1, 1)},
            ]
        ],
        "n": [3],
    },
    schema={
        "icds": pl.List(
            pl.Struct(
                {
                    "icd": pl.String,
                    "location": pl.Enum(["R", "L", "B"]),
                    "date": pl.Date,
                }
            )
        ),
        "n": pl.UInt32,
    },
)
print(df)
out = df.select(pl.col("icds").repeat_by(pl.col("n")))
with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000):
    print(out)

df = pl.DataFrame(
    {
        "icds": [
            [
                {
                    "icd": {"group": "A", "value": "123"},
                    "location": "L",
                },
                {
                    "icd": {"group": "B", "value": "456"},
                    "location": None,
                },
            ]
        ],
        "n": [3],
    },
)
print(df)
out = df.select(pl.col("icds").repeat_by(pl.col("n")))
with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000):
    print(out)

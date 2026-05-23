from yt_lib.yt_search import yt_search

results = yt_search(
    "Center of mass frame of reference",
    max_results=3,
)

for item in results["items"]:
    print(item["title"])
    print(item["url"])
    print()
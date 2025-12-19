import os
from icrawler.builtin import BingImageCrawler

birds = [
    "carrion crow",
    "wood pigeon",
    "magpie",
    "blue tit",
    "great tit",
    "robin",
    "house sparrow",
    "blackbird",
    "starling",
    "goldfinch",
    "ring-necked parakeet"
]

IMAGES_DIR = "images"

for bird in birds:
    folder = os.path.join(IMAGES_DIR, bird.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=f"{bird} bird real", max_num=200)

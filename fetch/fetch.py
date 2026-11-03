import json
import os

import requests

from text_utils import clean_html

TIMEOUT = 15

for path in ("postings_jobicy.json"):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)


def lever_description(posting) -> str:
    parts = [posting.get("description", "")]
    for section in posting.get("lists", []):
        parts.append(f"<h3>{section['text']}</h3><ul>{section['content']}</ul>")
    parts.append(posting.get("additional", ""))
    return clean_html("".join(parts))


def fetch_lever(slugs):
    lever_url = "https://api.lever.co/v0/postings/{slug}?mode=json"

    for slug in slugs:
        response = requests.get(lever_url.format(slug=slug), timeout=TIMEOUT)
        response.raise_for_status()
        for posting in response.json():
            posting_id = posting["id"]
            if posting_id in seen_ids:
                continue
            seen_ids.add(posting_id)
            postings_to_add.append({
                "id": posting_id,
                "title": posting["text"],
                "url": posting["hostedUrl"],
                "slug": slug,
                "location": posting["categories"].get("location"),
                "department": posting["categories"].get("department"),
                "posted_at": posting["createdAt"],
                "description": lever_description(posting),
            })


def fetch_jobicy():
    jobicy_url = "https://jobicy.com/api/v2/remote-jobs?count=20&tag=python"

    response = requests.get(jobicy_url, timeout=TIMEOUT)
    response.raise_for_status()
    for posting in response.json()["jobs"]:
        posting_id = f"jobicy:{posting['id']}"
        if posting_id in seen_ids:
            continue
        seen_ids.add(posting_id)
        postings_to_add.append({
            "id": posting_id,
            "title": posting["jobTitle"],
            "url": posting["url"],
            "slug": posting["companyName"],
            "location": posting["jobGeo"],
            "department": posting["jobIndustry"],
            "posted_at": posting["pubDate"],
            "description": clean_html(posting["jobDescription"]),
        })


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


slugs = []
seen_ids = set()
postings_to_add = []

# with open("companies.json") as c:
#     slugs = [company["slug"] for company in json.load(c) if company["ats"] == "lever"]
#
# with open("seen.json") as c:
#     seen_ids.update(json.load(c))
#
# fetch_lever(slugs)
# save("postings.json", postings_to_add)
# save("seen.json", sorted(seen_ids))

fetch_jobicy()
save("postings_jobicy.json", postings_to_add)

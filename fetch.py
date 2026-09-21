import json
import requests
import os

for path in ("seen.json", "postings.json", "companies.json"):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)

def fetch_lever(slugs):
    lever_url = "https://api.lever.co/v0/postings/{slug}?mode=json"

    for slug in slugs:
        response = requests.get(lever_url.format(slug=slug))
        for posting in response.json():
            if posting["id"] not in seen_ids:
                postings_to_add.append({
                    "id": posting["id"],
                    "title": posting["text"],
                    "url": posting["hostedUrl"],
                    "slug": slug,
                    "location": posting["categories"]["location"],
                    "department": posting["categories"]["department"],
                    "posted_at": posting["createdAt"],
                    "description": "\n ".join([x["text"] + "\n " + x["content"] for x in posting["lists"]]),
                })
                seen_ids.update(p["id"] for p in postings_to_add)


def fetch_jobicy():
    jobicy_url = "https://jobicy.com/api/v2/remote-jobs?count=20&tag=python"

    response = requests.get(jobicy_url)
    for posting in response.json()["jobs"]:
        if posting["id"] not in seen_ids:
            postings_to_add.append({
                "id": posting["id"],
                "title": posting["jobTitle"],
                "url": posting["url"],
                "slug": posting["companyName"],
                "location": posting["jobGeo"],
                "department": posting["jobIndustry"],
                "posted_at": posting["pubDate"],
                "description": posting["jobDescription"]
            })
            seen_ids.update(p["id"] for p in postings_to_add)


slugs = []
seen_ids = set()
postings_to_add = []

with open("companies.json") as c:
    companies = json.load(c)
    for company in companies:
        if company["ats"] == "lever":
            slugs.append(company["slug"])

with open("seen.json") as c:
    seen_ids.update(json.load(c))

fetch_lever(slugs)

with open("postings.json", "a") as f:
    json.dump(postings_to_add, f)

with open("seen.json", "w") as f:
    json.dump(sorted(seen_ids), f)
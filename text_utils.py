from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"]


def clean_html(text: str | None) -> str:
    """Turn an HTML job description into plain text, one block per line."""
    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    for tag in soup.find_all("li"):
        tag.insert(0, "- ")

    lines = (" ".join(line.split()) for line in soup.get_text().split("\n"))
    return "\n".join(line for line in lines if line)

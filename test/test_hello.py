from comic.hello import hello_comic_translate


def test_hello():
    assert hello_comic_translate() == "Hello, comic translate!"

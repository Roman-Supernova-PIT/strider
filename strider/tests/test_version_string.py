def test_version_is_string():
    from strider import __version__

    assert isinstance(__version__, str)

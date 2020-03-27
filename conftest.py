
def pytest_addoption(parser):
    parser.addoption("--supervised-url", action='store')


def pytest_generate_tests(metafunc):
    supervised_url = metafunc.config.option.supervised_url

    if 'supervised_url' in metafunc.fixturenames \
            and supervised_url is not None:
        metafunc.parametrize('supervised_url', [supervised_url])

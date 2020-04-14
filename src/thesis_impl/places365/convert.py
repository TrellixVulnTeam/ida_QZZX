from argparse import ArgumentParser

from thesis_impl.convert import main
from thesis_impl.places365.hub import Places365Hub


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert a subset of the '
                                        'Places365 data to a Parquet'
                                        'store.')
    main(parser, Places365Hub)

from argparse import ArgumentParser

from simadv.convert import main
from simadv.places365.hub import Places365Hub


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert a subset of the '
                                        'Places365 data to a Parquet'
                                        'store.')
    main(parser, Places365Hub)

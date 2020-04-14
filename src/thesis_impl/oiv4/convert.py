from argparse import ArgumentParser

from thesis_impl.convert import main
from thesis_impl.oiv4.hub import OpenImagesV4Hub


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert a subset of the '
                                        'OpenImages data to a Parquet'
                                        'store.')
    main(parser, OpenImagesV4Hub)

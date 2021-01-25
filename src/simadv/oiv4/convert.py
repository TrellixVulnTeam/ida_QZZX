from argparse import ArgumentParser

from simadv.convert import main
from simadv.oiv4.hub import OpenImagesV4Hub


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert a subset of the '
                                        'OpenImages data to a Parquet'
                                        'store.')
    main(parser, OpenImagesV4Hub)

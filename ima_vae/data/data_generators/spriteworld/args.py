import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nclasses", type=int, default=1, help="Number of auxiliary variables"
    )
    parser.add_argument(
        "--nobs", type=int, default=10000, help="Number of observations in dataset"
    )
    parser.add_argument(
        "--lower",
        type=int,
        default=2,
        help="Lower bound on alpha and beta (Set to at least 2)",
    )
    parser.add_argument(
        "--upper", type=int, default=15, help="Upper bound on alpha and beta"
    )
    parser.add_argument(
        "--angle", type=bool, default=False, help="True if you want angle as a factor"
    )
    parser.add_argument(
        "--shape", type=bool, default=False, help="True if you want shape as a factor"
    )
    parser.add_argument(
        "--projective",
        type=bool,
        default=False,
        help="True if you want to apply a projective transformation (to destroy colum-ortogonality)",
    )
    parser.add_argument(
        "--affine",
        type=bool,
        default=False,
        help="True if you want to apply an affine transformation (to destroy colum-ortogonality)",
    )
    parser.add_argument(
        "--deltah", type=int, default=0, help="Disturbance in the Hue channel"
    )
    parser.add_argument(
        "--deltas", type=int, default=0, help="Disturbance in the Saturation channel"
    )
    parser.add_argument(
        "--deltav", type=int, default=0, help="Disturbance in the Value channel"
    )
    args = parser.parse_args()

    return args

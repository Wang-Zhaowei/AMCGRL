import argparse


def settings():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inter_type', required=True,
                        help='Predicted type of the molecular interaction. e.g., NN, NA, or AA')

    parser.add_argument('--input_x_file', required=True,
                        help='Path to data result file. e.g., peptides.txt')

    parser.add_argument('--input_y_file', required=True,
                        help='Path to data result file. e.g., proteins.txt')
    
    parser.add_argument('--out_file', required=True,
                        help='Path to data result file. e.g., result.txt')

    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the output from each encoder. Default is 256.')

    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Weight of the attribute similarity. Default is 0.8.')

    parser.add_argument('--beta', type=float, default=0.6,
                        help='Weight of the attribute similarity (another molecular type). Default is 0.6.')

    parser.add_argument('--att_g', type=int, default=4,
                        help='Number of attention heads of MHGA. Default is 4.')

    parser.add_argument('--att_c', type=int, default=4,
                        help='Number of attention heads of MHCA. Default is 4.')

    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Initial learning rate. Default is 5e-3.')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train. Default is 100.')

    args = parser.parse_args()

    return args

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('-c', '--code', nargs='*', required=True, type=float, help='which code is used for controllable style generation.')
        parser.add_argument('-i', '--image_dir', type=str, help='where the source images are stored. Assume that labels are saved in the folder name by replacing "Images" to "Labels"')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--sw_batch_size', type=int, default=1, help='sub-batch size of sliding window inference.')
        parser.add_argument('--overlap', type=float, default=0.7, help='overlapping ratio of sliding window inference.')
        
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=20, help='how many test images to run')

        # playground
        parser.add_argument('--src_image', type=str, help='the path of a single source domain image.')
        parser.add_argument('--src_label', type=str, help='the path of the source domain label.')
        parser.add_argument('--save_dir',  type=str, help='the directory to save the output image for the playground.')

        self.isTrain = False
        return parser

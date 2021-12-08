import imageio as iio

class VideoMaker:
    """
    Helper class to turn RGB numpy arrays into videos,
    save videos and concatenate multiple videos
    """
    def __init__(self, write_path, read_path):
        if write_path is not None:
            self.writer = iio.get_writer(write_path, fps=30)
        if read_path is not None:
            self.reader = iio.get_reader(read_path, fps=30)

    def new_writer(self, write_path):
        """
        Init new writer at a new path
        """
        self.writer.close()
        self.writer = iio.get_writer(write_path, fps=30)
    
    def write_to_file(self, arr):
        """
            Helper function to write an array of RGB images as a video and
            append it to an already existing mp4 file. If the file does not
            exist, it will be created.
            Args:
                arr: List of numpy arrays, assumed to be (H, W, C)
        """
        for frame in arr:
            self.writer.append_data(frame)
    
    def close_writer(self):
        """
        Helper function to close writer
        after image is saved.
        """
        self.writer.close()
        
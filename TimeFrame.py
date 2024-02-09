class TimeFrame:
    begin_frame: int = 0
    end_frame: int = 0
    upper_bound: int = 0
    lower_bound: int = 0

    def __init__(self, begin_frame: int, end_frame: int, lower_bound: int, upper_bound: int):
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
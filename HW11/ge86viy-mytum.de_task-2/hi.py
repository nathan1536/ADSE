def __init__(self, lists_name, transform=None):
    self.state_old = []
    self.action = []
    # read data
    lists = pd.read_csv(lists_name)["file"].values.tolist()
    for d_ in lists:
        df = pd.DataFrame(pd.read_pickle(d_))
        s_ = df.iloc[:, 0].values.tolist()
        a_ = df.iloc[:, 1].values.tolist()
        self.state_old.extend(s_[:-1])
        self.action.extend(a_[:-1])
    self.state_old = np.array(self.state_old)
    self.action = np.array(self.action)
    self.n_data = len(self.state_old)
    self.s_dim = self.state_old[0].shape
    self.a_dim = self.action[0].shape
    if len(self.s_dim) == 1:
        self.dtype = np.float32
        self.transform = None
    else:
        self.dtype = np.uint8
        self.transform = transform
    print("number of data: {}".format(self.n_data))
    print("observation type is {}".format(self.dtype))

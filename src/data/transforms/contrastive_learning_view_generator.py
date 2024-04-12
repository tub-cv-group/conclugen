class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        if isinstance(base_transform, list):
            assert len(base_transform) == n_views, 'You need to pass as many '\
                'transforms in the list as there are views. '\
                f'Got num transforms {len(base_transform)} and num views {n_views}.'
            self.base_transform = base_transform
        else:
            # Create a list of copies of the base transform
            self.base_transform = [base_transform] * n_views
        self.n_views = n_views

    def __call__(self, x):
        imgs = [self.base_transform[i](x) for i in range(self.n_views)]
        return imgs
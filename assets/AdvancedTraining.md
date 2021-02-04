### Advanced usage of easytorch for:
* #### Initialize multiple optimizer/neural networks.
* #### Implement complex training and backprop. steps with them.
* #### Easily save predictions for any prediction task(Like segmentation results).

### Override the following:
```python
def _init_optimizer(self):
    r"""
    Initialize optimizers.
    """
    self.optimizer['adam'] = torch.optim.Adam(self.nn['model'].parameters(), lr=self.args['learning_rate'])

def training_iteration(self, batch):
    '''
    ### Optional
    If you need complex/mixed training steps, it can be done here. 
    If not, no need to extend this method 
    '''
    self.optimizer['adam'].zero_grad()
    it = self.iteration(batch)
    it['loss'].backward()
    self.optimizer['adam'].step()
    return it


def save_predictions(self, dataset, its):
    '''
    If one wants to save predictions(For example, segmentation result.)
    '''
    pass

 def new_metrics(self):
        r"""
        User can override to supply desired implementation of easytorch.core.metrics.ETMetrics().
            Example: easytorch.metrics.Prf1a() will work with precision, recall, F1, Accuracy, IOU scores.
        """
        return easytorch.Prf1a()

def new_averages(self):
    r""""
    Should supply an implementation of easytorch.metrics.ETAverages() that can keep track of multiple averages.
        Example: multiple loss, or any other values.
    """
    return easytorch.ETAverages(num_averages=1)

def init_experiment_cache(self):
    r"""
    An extra layer to reset cache for each dataspec. For example:
    1. Set a new score to monitor:
        self.cache['monitor_metric'] = 'Precision'
        self.cache['metric_direction'] = 'maximize'
                        OR
        self.cache['monitor_metric'] = 'MSE'
        self.cache['metric_direction'] = 'minimize'
                        OR
                to save latest model
        self.cache['monitor_metric'] = 'time'
        self.cache['metric_direction'] = 'maximize'
    2. Set new log_headers based on what is returned by get() method
        of your implementation of easytorch.metrics.ETMetrics and easytorch.metrics.ETAverages class:
        For example, the default implementation is:
        - The get method of easytorch.metrics.ETAverages class returns the average loss value.
        - The get method of easytorch.metrics.Prf1a returns Precision,Recall,F1,Accuracy
        - so Default heade is [Loss,Precision,Recall,F1,Accuracy]
    3. Set new log_dir based on different experiment versions on each datasets as per info. received from arguments.
    """
    pass
```

### Examples:
#### * **[DCGAN Example.](https://github.com/sraashis/gan-easytorch-celeb-faces)**
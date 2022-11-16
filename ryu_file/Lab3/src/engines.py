from urllib.parse import quote_plus
from numpy import dtype
import torch

from torchmetrics.aggregation import MeanMetric

def pretrain_baseline(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


def train_baseline(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        targets = targets.reshape((-1,))
        print(outputs.shape)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


def evaluate_baseline(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        print(len(loader))
        print(f'outputs 의 크기 : {outputs.shape}')
        print(f'targets 의 크기 : {targets.shape}')

        outputs = outputs.reshape((-1, outputs.shape[-1]))
        targets = targets.reshape((-1))
        print(f'outputs 의 크기 : {outputs.shape}')
        print(f'targets 의 크기 : {targets.shape}')
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary
    



def train_prototype(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for support_inputs, query_inputs, query_targets in loader:
        support_inputs = support_inputs.to(device)
        query_inputs = query_inputs.to(device)
        query_targets = query_targets.to(device)

        query_outputs = model(support_inputs, query_inputs)
        query_outputs = query_outputs.to(device)

        query_outputs = query_outputs.reshape((-1, query_outputs.shape[-1]))
        query_targets = query_targets.reshape((-1,))

        #print(f'query_outputs.shape : {query_outputs.shape}')
        #print(f'queyry_targets.shape : {query_targets.shape}')

        loss = loss_fn(query_outputs, query_targets)
        metric = metric_fn(query_outputs, query_targets)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


def evaluate_prototype(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for support_inputs, query_inputs, query_targets in loader:
        support_inputs = support_inputs.to(device)
        query_inputs = query_inputs.to(device)
        query_targets = query_targets.to(device)
        
        query_outputs = model(support_inputs, query_inputs)
        query_outputs = query_outputs.reshape((-1, query_outputs.shape[-1]))
        query_targets = query_targets.reshape((-1,))
        loss = loss_fn(query_outputs, query_targets)
        metric = metric_fn(query_outputs, query_targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary
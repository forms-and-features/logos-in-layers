#!/usr/bin/env python3
"""CPU-only tests for hooks helpers with minimal mocks."""

import torch

from layers_core.hooks import build_cache_hook, attach_residual_hooks, detach_hooks


class MockHandle:
    def __init__(self, hookpoint):
        self.hp = hookpoint
    def remove(self):
        self.hp._fn = None


class MockHookPoint:
    def __init__(self, name):
        self.name = name
        self._fn = None
    def add_hook(self, fn):
        self._fn = fn
        return MockHandle(self)
    # Utility for tests to simulate firing
    def fire(self, tensor):
        class HookObj:
            pass
        h = HookObj()
        h.name = self.name
        self._fn(tensor, h)


class MockBlock:
    def __init__(self, idx):
        self.hook_resid_post = MockHookPoint(f"blocks.{idx}.hook_resid_post")


class MockModel:
    class Cfg:
        def __init__(self, n):
            self.n_layers = n

    def __init__(self, n_layers=2, with_pos=True):
        self.cfg = self.Cfg(n_layers)
        self.blocks = [MockBlock(i) for i in range(n_layers)]
        self.hook_dict = {
            'hook_embed': MockHookPoint('hook_embed')
        }
        if with_pos:
            self.hook_dict['hook_pos_embed'] = MockHookPoint('hook_pos_embed')


def test_attach_and_cache_with_positional():
    model = MockModel(n_layers=2, with_pos=True)
    cache = {}
    cache_hook = build_cache_hook(cache)
    handles, has_pos = attach_residual_hooks(model, cache_hook)
    assert has_pos is True
    assert len(handles) == 1 + 1 + model.cfg.n_layers  # embed + pos + per-layer

    # Fire some hooks and ensure cache is populated with correct keys
    x = torch.randn(1, 3)
    model.hook_dict['hook_embed'].fire(x)
    model.hook_dict['hook_pos_embed'].fire(x)
    model.blocks[0].hook_resid_post.fire(x)
    assert 'hook_embed' in cache
    assert 'hook_pos_embed' in cache
    assert 'blocks.0.hook_resid_post' in cache

    detach_hooks(handles)
    # After detach, firing should not change cache
    before = dict(cache)
    model.hook_dict['hook_embed'].fire(x)
    assert cache == before


def test_attach_without_positional():
    model = MockModel(n_layers=1, with_pos=False)
    cache = {}
    cache_hook = build_cache_hook(cache)
    handles, has_pos = attach_residual_hooks(model, cache_hook)
    assert has_pos is False
    assert len(handles) == 1 + model.cfg.n_layers


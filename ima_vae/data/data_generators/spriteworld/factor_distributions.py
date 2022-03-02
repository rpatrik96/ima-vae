from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import numpy as np
import torch
from torch.distributions.beta import Beta as beta_dist
import six
from scipy.stats import beta


@six.add_metaclass(abc.ABCMeta)
class AbstractDistribution(object):
    """Abstract class from which all distributions should inherit."""

    @abc.abstractmethod
    def sample(self, rng=None):
        """Sample a spec from this distribution. Returns a dictionary.
        Args:
          rng: Random number generator. Fed into self._get_rng(), if None defaults
            to np.random.
        """

    @abc.abstractmethod
    def contains(self, spec):
        """Return whether distribution contains spec dictionary."""

    @abc.abstractmethod
    def to_str(self, indent):
        """Recursive string description of this distribution."""

    def __str__(self):
        return self.to_str(indent=0)

    def _get_rng(self, rng=None):
        """Get random number generator, defaulting to np.random."""
        return np.random if rng is None else rng

    @abc.abstractproperty
    def keys(self):
        """The set of keys in specs sampled from this distribution."""


class Continuous(AbstractDistribution):
    """Continuous 1-dimensional uniform distribution."""

    def __init__(self, key, minval, maxval, dtype="float32"):
        """Construct continuous 1-dimensional uniform distribution.
        Args:
          key: String factor name. self.sample() returns {key: _}.
          minval: Scalar minimum value.
          maxval: Scalar maximum value.
          dtype: String numpy dtype.
        """
        self.key = key
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

    def sample(self, uniform, rng=None):
        """Sample value in [self.minval, self.maxval) and return dict."""
        rng = self._get_rng(rng)
        out = rng.uniform(low=self.minval, high=self.maxval)
        out = np.cast[self.dtype](out)
        return {self.key: out}

    def contains(self, spec):
        """Check if spec[self.key] is in [self.minval, self.maxval)."""
        if self.key not in spec:
            raise KeyError(
                "key {} is not in spec {}, but must be to evaluate "
                "containment.".format(self.key, spec)
            )
        else:
            return spec[self.key] >= self.minval and spec[self.key] < self.maxval

    def to_str(self, indent):
        s = "<Continuous: key={}, mival={}, maxval={}, dtype={}>".format(
            self.key, self.minval, self.maxval, self.dtype
        )
        return indent * "  " + s

    @property
    def keys(self):
        return set([self.key])


class Beta(AbstractDistribution):
    """Continuous 1-dimensional beta distribution."""

    def __init__(self, key, alpha, beta, independent=True, dtype="float32"):
        self.key = key
        self.alpha = alpha
        self.beta = beta
        self.independent = independent
        self.dtype = dtype

    def sample(self, uniform, rng=None):
        if self.independent == True:
            out = beta_dist(
                torch.tensor([self.alpha]), torch.tensor([self.beta])
            ).sample()
            if self.key == "x":
                out = out * 0.55 + 0.2
            elif self.key == "y":
                out = out * 0.55 + 0.2
            elif self.key == "scale":
                out = out * 0.09 + 0.11
            elif self.key == "angle":
                out *= 360
        else:
            if self.key == "x":
                out = beta.ppf(uniform[:, 0], self.alpha, self.beta)
                out = out * 0.55 + 0.2
            elif self.key == "y":
                out = beta.ppf(uniform[:, 1], self.alpha, self.beta)
                out = out * 0.55 + 0.2
            elif self.key == "scale":
                out = beta.ppf(uniform[:, 2], self.alpha, self.beta)
                out = out * 0.09 + 0.11
            elif self.key == "c0":
                out = beta.ppf(uniform[:, 3], self.alpha, self.beta)
            elif self.key == "angle":
                out = beta.ppf(uniform[:, 4], self.alpha, self.beta)
                out *= 360

        out = np.cast[self.dtype](out)
        return {self.key: out}

    def contains(self, spec):
        return True

    def to_str(self, indent):
        s = "<Continuous: key={}, alpha={}, beta={}, dtype={}>".format(
            self.key, self.alpha, self.beta, self.dtype
        )
        return indent * "  " + s

    @property
    def keys(self):
        return set([self.key])


class Discrete(AbstractDistribution):
    """Discrete distribution."""

    def __init__(self, key, candidates, probs=None):
        """Construct discrete distribution.
        Args:
          key: String. Factor name.
          candidates: Iterable. Discrete values to sample from.
          probs: None or iterable of floats summing to 1. Candidate sampling
            probabilities. If None, candidates are sampled uniformly.
        """
        self.candidates = candidates
        self.key = key
        self.probs = probs

    def sample(self, uniform, rng=None):
        rng = self._get_rng(rng)
        out = self.candidates[rng.choice(len(self.candidates), p=self.probs)]
        return {self.key: out}

    def contains(self, spec):
        if self.key not in spec:
            raise KeyError(
                "key {} is not in spec {}, but must be to evaluate "
                "containment.".format(self.key, spec)
            )
        else:
            return spec[self.key] in self.candidates

    def to_str(self, indent):
        s = "<Discrete: key={}, candidates={}, probs={}>".format(
            self.key, self.candidates, self.probs
        )
        return indent * "  " + s

    @property
    def keys(self):
        return set([self.key])


class Product(AbstractDistribution):
    """Product distribution."""

    def __init__(self, components):
        """Construct product distribution.
        This is used to create distributions over larger numbers of factors by
        taking the product of components. The components must have disjoint key
        sets.
        Args:
          components: Iterable of distributions.
        """
        self.components = components

        self._keys = functools.reduce(set.union, [set(c.keys) for c in components])
        num_keys = sum(len(c.keys) for c in components)
        if len(self._keys) < num_keys:
            raise ValueError(
                "All components must have different keys, yet there are {} "
                "overlapping keys.".format(num_keys - len(self._keys))
            )

    def sample(self, uniform, rng=None):
        rng = self._get_rng(rng)
        sample = {}
        for c in self.components:
            sample.update(c.sample(uniform, rng=rng))
        return sample

    def contains(self, spec):
        return all(c.contains(spec) for c in self.components)

    def to_str(self, indent):
        components_strings = [x.to_str(indent + 2) for x in self.components]
        s = (
            indent * "  "
            + "<Product:\n"
            + (indent + 1) * "  "
            + "components=[\n{},\n"
            + (indent + 1) * "  "
            + "]>"
        ).format(",\n".join(components_strings))
        return s

    @property
    def keys(self):
        return self._keys

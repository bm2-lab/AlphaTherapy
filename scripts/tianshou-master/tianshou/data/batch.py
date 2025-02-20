import torch
import pprint
import warnings
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Any, List, Tuple, Union, Iterator, Optional

# Disable pickle warning related to torch, since it has been removed
# on torch master branch. See Pull Request #39003 for details:
# https://github.com/pytorch/pytorch/pull/39003
warnings.filterwarnings(
    "ignore", message="pickle support for Storage will be removed in 1.5.")


def _is_batch_set(data: Any) -> bool:
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and all(isinstance(e, (dict, Batch)) for e in data):
            return True
    elif isinstance(data, np.ndarray) and data.dtype == object:
        if all(isinstance(e, (dict, Batch)) for e in data.tolist()):
            return True
    return False


def _create_value(inst: Any, size: int, stack=True) -> Union[
        'Batch', np.ndarray, torch.Tensor]:
    """
    :param bool stack: whether to stack or to concatenate. E.g. if inst has
        shape of (3, 5), size = 10, stack=True returns an np.ndarry with shape
        of (10, 3, 5), otherwise (10, 5)
    """
    has_shape = isinstance(inst, (np.ndarray, torch.Tensor))
    is_scalar = \
        isinstance(inst, Number) or \
        issubclass(inst.__class__, np.generic) or \
        (has_shape and not inst.shape)
    if not stack and is_scalar:
        # here we do not consider scalar types, following the
        # behavior of numpy which does not support concatenation
        # of zero-dimensional arrays (scalars)
        raise TypeError(f"cannot cat {inst} with which is scalar")
    if has_shape:
        shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        if issubclass(inst.dtype.type, (np.bool_, np.number)):
            target_type = inst.dtype.type
        else:
            target_type = object
        return np.full(shape,
                       fill_value=None if target_type == object else 0,
                       dtype=target_type)
    elif isinstance(inst, torch.Tensor):
        return torch.full(shape,
                          fill_value=0,
                          device=inst.device,
                          dtype=inst.dtype)
    elif isinstance(inst, (dict, Batch)):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = _create_value(val, size, stack=stack)
        return zero_batch
    elif is_scalar:
        return _create_value(np.asarray(inst), size, stack=stack)
    else:  # fall back to object
        return np.array([None for _ in range(size)])


def _assert_type_keys(keys):
    keys = list(keys)
    assert all(isinstance(e, str) for e in keys), \
        f"keys should all be string, but got {keys}"


class Batch:
    """Tianshou provides :class:`~tianshou.data.Batch` as the internal data
    structure to pass any kind of data to other methods, for example, a
    collector gives a :class:`~tianshou.data.Batch` to policy for learning.
    Here is the usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import Batch
        >>> data = Batch(a=4, b=[5, 5], c='2312312')
        >>> # the list will automatically be converted to numpy array
        >>> data.b
        array([5, 5])
        >>> data.b = np.array([3, 4, 5])
        >>> print(data)
        Batch(
            a: 4,
            b: array([3, 4, 5]),
            c: '2312312',
        )

    In short, you can define a :class:`Batch` with any key-value pair.

    For Numpy arrays, only data types with ``object``, bool, and number
    are supported. For strings or other data types, however, they can be
    held in ``object`` arrays.

    The current implementation of Tianshou typically use 7 reserved keys in
    :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()``\
        function returns 4 arguments, and the last one is ``info``);
    * ``policy`` the data computed by policy in step :math:`t`;

    :class:`~tianshou.data.Batch` object can be initialized by a wide variety
    of arguments, ranging from the key/value pairs or dictionary, to list and
    Numpy arrays of :class:`dict` or Batch instances where each element is
    considered as an individual sample and get stacked together:
    ::

        >>> data = Batch([{'a': {'b': [0.0, "info"]}}])
        >>> print(data[0])
        Batch(
            a: Batch(
                b: array([0.0, 'info'], dtype=object),
            ),
        )

    :class:`~tianshou.data.Batch` has the same API as a native Python
    :class:`dict`. In this regard, one can access stored data using string
    key, or iterate over stored data:
    ::

        >>> data = Batch(a=4, b=[5, 5])
        >>> print(data["a"])
        4
        >>> for key, value in data.items():
        >>>     print(f"{key}: {value}")
        a: 4
        b: [5, 5]


    :class:`~tianshou.data.Batch` also partially reproduces the Numpy API for
    arrays. It also supports the advanced slicing method, such as batch[:, i],
    if the index is valid. You can access or iterate over the individual
    samples, if any:
    ::

        >>> data = Batch(a=np.array([[0.0, 2.0], [1.0, 3.0]]), b=[[5, -5]])
        >>> print(data[0])
        Batch(
            a: array([0., 2.])
            b: array([ 5, -5]),
        )
        >>> for sample in data:
        >>>     print(sample.a)
        [0., 2.]

        >>> print(data.shape)
        [1, 2]
        >>> data[:, 1] += 1
        >>> print(data)
        Batch(
            a: array([[0., 3.],
                      [1., 4.]]),
            b: array([[ 5, -4]]),
        )

    Similarly, one can also perform simple algebra on it, and stack, split or
    concatenate multiple instances:
    ::

        >>> data_1 = Batch(a=np.array([0.0, 2.0]), b=5)
        >>> data_2 = Batch(a=np.array([1.0, 3.0]), b=-5)
        >>> data = Batch.stack((data_1, data_2))
        >>> print(data)
        Batch(
            b: array([ 5, -5]),
            a: array([[0., 2.],
                      [1., 3.]]),
        )
        >>> print(np.mean(data))
        Batch(
            b: 0.0,
            a: array([0.5, 2.5]),
        )
        >>> data_split = list(data.split(1, False))
        >>> print(list(data.split(1, False)))
        [Batch(
            b: array([5]),
            a: array([[0., 2.]]),
        ), Batch(
            b: array([-5]),
            a: array([[1., 3.]]),
        )]
        >>> data_cat = Batch.cat(data_split)
        >>> print(data_cat)
        Batch(
            b: array([ 5, -5]),
            a: array([[0., 2.],
                      [1., 3.]]),
        )

    Note that stacking of inconsistent data is also supported. In which case,
    ``None`` is added in list or :class:`np.ndarray` of objects, 0 otherwise.
    ::

        >>> data_1 = Batch(a=np.array([0.0, 2.0]))
        >>> data_2 = Batch(a=np.array([1.0, 3.0]), b='done')
        >>> data = Batch.stack((data_1, data_2))
        >>> print(data)
        Batch(
            a: array([[0., 2.],
                      [1., 3.]]),
            b: array([None, 'done'], dtype=object),
        )

    Method ``empty_`` sets elements to 0 or ``None`` for ``object``.
    ::

        >>> data.empty_()
        >>> print(data)
        Batch(
            a: array([[0., 0.],
                      [0., 0.]]),
            b: array([None, None], dtype=object),
        )
        >>> data = Batch(a=[False,  True], b={'c': [2., 'st'], 'd': [1., 0.]})
        >>> data[0] = Batch.empty(data[1])
        >>> data
        Batch(
            a: array([False,  True]),
            b: Batch(
                   c: array([None, 'st']),
                   d: array([0., 0.]),
               ),
        )

    :meth:`~tianshou.data.Batch.shape` and :meth:`~tianshou.data.Batch.__len__`
    methods are also provided to respectively get the shape and the length of
    a :class:`Batch` instance. It mimics the Numpy API for Numpy arrays, which
    means that getting the length of a scalar Batch raises an exception.
    ::

        >>> data = Batch(a=[5., 4.], b=np.zeros((2, 3, 4)))
        >>> data.shape
        [2]
        >>> len(data)
        2
        >>> data[0].shape
        []
        >>> len(data[0])
        TypeError: Object of type 'Batch' has no len()

    Convenience helpers are available to convert in-place the stored data into
    Numpy arrays or Torch tensors.

    Finally, note that :class:`~tianshou.data.Batch` is serializable and
    therefore Pickle compatible. This is especially important for distributed
    sampling.
    """

    def __init__(self,
                 batch_dict: Optional[Union[
                     dict, 'Batch', Tuple[Union[dict, 'Batch']],
                     List[Union[dict, 'Batch']], np.ndarray]] = None,
                 copy: bool = False,
                 **kwargs) -> None:
        if copy:
            batch_dict = deepcopy(batch_dict)
        if batch_dict is not None:
            if isinstance(batch_dict, (dict, Batch)):
                _assert_type_keys(batch_dict.keys())
                for k, v in batch_dict.items():
                    if isinstance(v, (list, tuple, np.ndarray)):
                        v_ = None
                        if not isinstance(v, np.ndarray) and \
                                all(isinstance(e, torch.Tensor) for e in v):
                            self.__dict__[k] = torch.stack(v)
                            continue
                        else:
                            v_ = np.asanyarray(v)
                        if v_.dtype != object:
                            v = v_  # normal data list, this is the main case
                            if not issubclass(v.dtype.type,
                                              (np.bool_, np.number)):
                                v = v.astype(object)
                        else:
                            if _is_batch_set(v):
                                v = Batch(v)  # list of dict / Batch
                            else:
                                # this is actually a data list with objects
                                v = v_
                        self.__dict__[k] = v
                    elif isinstance(v, dict):
                        self.__dict__[k] = Batch(v)
                    else:
                        self.__dict__[k] = v
            elif _is_batch_set(batch_dict):
                self.stack_(batch_dict)
        if len(kwargs) > 0:
            self.__init__(kwargs, copy=copy)

    def __setattr__(self, key: str, value: Any):
        """self[key] = value"""
        if isinstance(value, list):
            if _is_batch_set(value):
                value = Batch(value)
            else:
                value = np.array(value)
                if not issubclass(value.dtype.type, (np.bool_, np.number)):
                    value = value.astype(object)
        elif isinstance(value, dict) or isinstance(value, np.ndarray) \
                and value.dtype == object and _is_batch_set(value):
            value = Batch(value)
        self.__dict__[key] = value

    def __getstate__(self):
        """Pickling interface. Only the actual data are serialized for both
        efficiency and simplicity.
        """
        state = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state):
        """Unpickling interface. At this point, self is an empty Batch instance
        that has not been initialized, so it can safely be initialized by the
        pickle state.
        """
        self.__init__(**state)

    def __getitem__(self, index: Union[
            str, slice, int, np.integer, np.ndarray, List[int]]) -> 'Batch':
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch_items = self.items()
        if len(batch_items) > 0:
            b = Batch()
            for k, v in batch_items:
                if isinstance(v, Batch) and len(v.__dict__) == 0:
                    b.__dict__[k] = Batch()
                else:
                    b.__dict__[k] = v[index]
            return b
        else:
            raise IndexError("Cannot access item from empty Batch object.")

    def __setitem__(self, index: Union[
            str, slice, int, np.integer, np.ndarray, List[int]],
            value: Any) -> None:
        """Assign value to self[index]."""
        if isinstance(value, np.ndarray):
            if not issubclass(value.dtype.type, (np.bool_, np.number)):
                value = value.astype(object)
        if isinstance(index, str):
            self.__dict__[index] = value
            return
        if not isinstance(value, (dict, Batch)):
            raise TypeError("Batch does not supported value type "
                            f"{type(value)} for item assignment.")
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise KeyError(
                "Creating keys is not supported by item assignment.")
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, np.ndarray) and \
                        issubclass(val.dtype.type, (np.bool_, np.number)):
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Union['Batch', Number, np.number]):
        """Algebraic addition with another :class:`~tianshou.data.Batch`
        instance in-place."""
        if isinstance(other, Batch):
            for (k, r), v in zip(self.__dict__.items(),
                                 other.__dict__.values()):
                # TODO are keys consistent?
                if r is None:
                    continue
                else:
                    self.__dict__[k] += v
            return self
        elif isinstance(other, (Number, np.number)):
            for k, r in self.items():
                if r is None:
                    continue
                else:
                    self.__dict__[k] += other
            return self
        else:
            raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, other: Union['Batch', Number, np.number]):
        """Algebraic addition with another :class:`~tianshou.data.Batch`
        instance out-of-place."""
        return deepcopy(self).__iadd__(other)

    def __imul__(self, val: Union[Number, np.number]):
        """Algebraic multiplication with a scalar value in-place."""
        assert isinstance(val, (Number, np.number)), \
            "Only multiplication by a number is supported."
        for k in self.__dict__.keys():
            self.__dict__[k] *= val
        return self

    def __mul__(self, val: Union[Number, np.number]):
        """Algebraic multiplication with a scalar value out-of-place."""
        return deepcopy(self).__imul__(val)

    def __itruediv__(self, val: Union[Number, np.number]):
        """Algebraic division with a scalar value in-place."""
        assert isinstance(val, (Number, np.number)), \
            "Only division by a number is supported."
        for k in self.__dict__.keys():
            self.__dict__[k] /= val
        return self

    def __truediv__(self, val: Union[Number, np.number]):
        """Algebraic division with a scalar value out-of-place."""
        return deepcopy(self).__itruediv__(val)

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k, v in self.items():
            rpl = '\n' + ' ' * (6 + len(k))
            obj = pprint.pformat(v).replace('\n', rpl)
            s += f'    {k}: {obj},\n'
            flag = True
        if flag:
            s += ')'
        else:
            s = self.__class__.__name__ + '()'
        return s

    def keys(self) -> List[str]:
        """Return self.keys()."""
        return self.__dict__.keys()

    def values(self) -> List[Any]:
        """Return self.values()."""
        return self.__dict__.values()

    def items(self) -> List[Tuple[str, Any]]:
        """Return self.items()."""
        return self.__dict__.items()

    def get(self, k: str, d: Optional[Any] = None) -> Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        return self.__dict__.get(k, d)

    def to_numpy(self) -> None:
        """Change all torch.Tensor to numpy.ndarray. This is an in-place
        operation.
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.detach().cpu().numpy()
            elif isinstance(v, Batch):
                v.to_numpy()

    def to_torch(self, dtype: Optional[torch.dtype] = None,
                 device: Union[str, int, torch.device] = 'cpu') -> None:
        """Change all numpy.ndarray to torch.Tensor. This is an in-place
        operation.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        for k, v in self.items():
            if isinstance(v, (np.number, np.bool_, Number, np.ndarray)):
                if isinstance(v, (np.number, np.bool_, Number)):
                    v = np.asanyarray(v)
                v = torch.from_numpy(v).to(device)
                if dtype is not None:
                    v = v.type(dtype)
                self.__dict__[k] = v
            elif isinstance(v, torch.Tensor):
                if dtype is not None and v.dtype != dtype or \
                        v.device.type != device.type or \
                        device.index is not None and \
                        device.index != v.device.index:
                    if dtype is not None:
                        v = v.type(dtype)
                    self.__dict__[k] = v.to(device)
            elif isinstance(v, Batch):
                v.to_torch(dtype, device)

    def cat_(self,
             batches: Union['Batch', List[Union[dict, 'Batch']]]) -> None:
        """Concatenate a list of (or one) :class:`~tianshou.data.Batch` objects
        into current batch.
        """
        if isinstance(batches, Batch):
            batches = [batches]
        if len(batches) == 0:
            return
        batches = [x if isinstance(x, Batch) else Batch(x) for x in batches]
        if len(self.__dict__) > 0:
            batches = [self] + list(batches)
        # partial keys will be padded by zeros
        # with the shape of [len, rest_shape]
        lens = [len(x) for x in batches]
        sum_lens = [0]
        for x in lens:
            sum_lens.append(sum_lens[-1] + x)
        keys_map = list(map(lambda e: set(e.keys()), batches))
        keys_shared = set.intersection(*keys_map)
        values_shared = [[e[k] for e in batches] for k in keys_shared]
        _assert_type_keys(keys_shared)
        for k, v in zip(keys_shared, values_shared):
            if all(isinstance(e, (dict, Batch)) for e in v):
                self.__dict__[k] = Batch.cat(v)
            elif all(isinstance(e, torch.Tensor) for e in v):
                self.__dict__[k] = torch.cat(v)
            else:
                v = np.concatenate(v)
                if not issubclass(v.dtype.type, (np.bool_, np.number)):
                    v = v.astype(object)
                self.__dict__[k] = v
        keys_partial = set.union(*keys_map) - keys_shared
        _assert_type_keys(keys_partial)
        for k in keys_partial:
            for i, e in enumerate(batches):
                val = e.get(k, None)
                if val is not None:
                    try:
                        self.__dict__[k][sum_lens[i]:sum_lens[i + 1]] = val
                    except KeyError:
                        self.__dict__[k] = \
                            _create_value(val, sum_lens[-1], stack=False)
                        self.__dict__[k][sum_lens[i]:sum_lens[i + 1]] = val

    @staticmethod
    def cat(batches: List[Union[dict, 'Batch']]) -> 'Batch':
        """Concatenate a list of :class:`~tianshou.data.Batch` object into a
        single new batch. For keys that are not shared across all batches,
        batches that do not have these keys will be padded by zeros with
        appropriate shapes. E.g.
        ::

            >>> a = Batch(a=np.zeros([3, 4]), common=Batch(c=np.zeros([3, 5])))
            >>> b = Batch(b=np.zeros([4, 3]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.cat([a, b])
            >>> c.a.shape
            (7, 4)
            >>> c.b.shape
            (7, 3)
            >>> c.common.c.shape
            (7, 5)
        """
        batch = Batch()
        batch.cat_(batches)
        return batch

    def stack_(self,
               batches: List[Union[dict, 'Batch']],
               axis: int = 0) -> None:
        """Stack a list of :class:`~tianshou.data.Batch` object into current
        batch.
        """
        if len(batches) == 0:
            return
        batches = [x if isinstance(x, Batch) else Batch(x) for x in batches]
        if len(self.__dict__) > 0:
            batches = [self] + list(batches)
        keys_map = list(map(lambda e: set(e.keys()), batches))
        keys_shared = set.intersection(*keys_map)
        values_shared = [[e[k] for e in batches] for k in keys_shared]
        _assert_type_keys(keys_shared)
        for k, v in zip(keys_shared, values_shared):
            if all(isinstance(e, (dict, Batch)) for e in v):
                self.__dict__[k] = Batch.stack(v, axis)
            elif all(isinstance(e, torch.Tensor) for e in v):
                self.__dict__[k] = torch.stack(v, axis)
            else:
                v = np.stack(v, axis)
                if not issubclass(v.dtype.type, (np.bool_, np.number)):
                    v = v.astype(object)
                self.__dict__[k] = v
        keys_partial = set.difference(set.union(*keys_map), keys_shared)
        if keys_partial and axis != 0:
            raise ValueError(
                f"Stack of Batch with non-shared keys {keys_partial} "
                f"is only supported with axis=0, but got axis={axis}!")
        _assert_type_keys(keys_partial)
        for k in keys_partial:
            for i, e in enumerate(batches):
                val = e.get(k, None)
                if val is not None:
                    try:
                        self.__dict__[k][i] = val
                    except KeyError:
                        self.__dict__[k] = \
                            _create_value(val, len(batches))
                        self.__dict__[k][i] = val

    @staticmethod
    def stack(batches: List[Union[dict, 'Batch']], axis: int = 0) -> 'Batch':
        """Stack a list of :class:`~tianshou.data.Batch` object into a single
        new batch. For keys that are not shared across all batches,
        batches that do not have these keys will be padded by zeros. E.g.
        ::

            >>> a = Batch(a=np.zeros([4, 4]), common=Batch(c=np.zeros([4, 5])))
            >>> b = Batch(b=np.zeros([4, 6]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.stack([a, b])
            >>> c.a.shape
            (2, 4, 4)
            >>> c.b.shape
            (2, 4, 6)
            >>> c.common.c.shape
            (2, 4, 5)

        .. note::

            If there are keys that are not shared across all batches, ``stack``
            with ``axis != 0`` is undefined, and will cause an exception.
        """
        batch = Batch()
        batch.stack_(batches, axis)
        return batch

    def empty_(self, index: Union[
        str, slice, int, np.integer, np.ndarray, List[int]] = None
    ) -> 'Batch':
        """Return an empty a :class:`~tianshou.data.Batch` object with 0 or
        ``None`` filled. If ``index`` is specified, it will only reset the
        specific indexed-data.
        """
        for k, v in self.items():
            if v is None:
                continue
            if isinstance(v, Batch):
                self.__dict__[k].empty_(index=index)
            elif isinstance(v, torch.Tensor):
                self.__dict__[k][index] = 0
            elif isinstance(v, np.ndarray):
                if v.dtype == object:
                    self.__dict__[k][index] = None
                else:
                    self.__dict__[k][index] = 0
            else:  # scalar value
                warnings.warn('You are calling Batch.empty on a NumPy scalar, '
                              'which may cause undefined behaviors.')
                if isinstance(v, (np.number, np.bool_, Number)):
                    self.__dict__[k] = v.__class__(0)
                else:
                    self.__dict__[k] = None
        return self

    @staticmethod
    def empty(batch: 'Batch', index: Union[
        str, slice, int, np.integer, np.ndarray, List[int]] = None
    ) -> 'Batch':
        """Return an empty :class:`~tianshou.data.Batch` object with 0 or
        ``None`` filled, the shape is the same as the given
        :class:`~tianshou.data.Batch`.
        """
        return deepcopy(batch).empty_(index)

    def update(self, batch: Optional[Union[dict, 'Batch']] = None,
               **kwargs) -> None:
        """Update this batch from another dict/Batch."""
        if batch is None:
            self.update(kwargs)
            return
        if isinstance(batch, dict):
            batch = Batch(batch)
        for k, v in batch.items():
            self.__dict__[k] = v
        if kwargs:
            self.update(kwargs)

    def __len__(self) -> int:
        """Return len(self)."""
        r = []
        for v in self.__dict__.values():
            if isinstance(v, Batch) and v.is_empty():
                continue
            elif hasattr(v, '__len__') and (not isinstance(
                    v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                r.append(len(v))
            else:
                raise TypeError("Object of type 'Batch' has no len()")
        if len(r) == 0:
            raise TypeError("Object of type 'Batch' has no len()")
        return min(r)

    def is_empty(self):
        return not any(
            not x.is_empty() if isinstance(x, Batch)
            else hasattr(x, '__len__') and len(x) > 0 for x in self.values())

    @property
    def shape(self) -> List[int]:
        """Return self.shape."""
        if len(self.__dict__.keys()) == 0:
            return []
        else:
            data_shape = []
            for v in self.__dict__.values():
                try:
                    data_shape.append(v.shape)
                except AttributeError:
                    raise TypeError("No support for 'shape' method with "
                                    f"type {type(v)} in class Batch.")
            return list(map(min, zip(*data_shape))) if len(data_shape) > 1 \
                else data_shape[0]

    def split(self, size: Optional[int] = None,
              shuffle: bool = True) -> Iterator['Batch']:
        """Split whole data into multiple small batches.

        :param int size: if it is ``None``, it does not split the data batch;
            otherwise it will divide the data batch with the given size.
            Default to ``None``.
        :param bool shuffle: randomly shuffle the entire data batch if it is
            ``True``, otherwise remain in the same. Default to ``True``.
        """
        length = len(self)
        if size is None:
            size = length
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        for idx in np.arange(0, length, size):
            yield self[indices[idx:(idx + size)]]

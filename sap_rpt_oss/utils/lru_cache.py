# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

class LRU_Cache:
    """
    This is just a remake of functools.lru_cache without the function call.
    """

    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.full = False
        self.cache: dict = {}
        self.hits = self.misses = 0
        self.root: list = []  # root of the circular doubly linked list
        self.root[:] = [
            self.root,
            self.root,
            None,
            None,
        ]  # initialize by pointing to self

    def __getitem__(self, key):
        link = self.cache.get(key)
        if link is None:
            self.misses += 1
            return None
        # Move the link to the front of the circular queue
        link_prev, link_next, _, result = link
        link_prev[self.NEXT] = link_next
        link_next[self.PREV] = link_prev
        last = self.root[self.PREV]
        last[self.NEXT] = self.root[self.PREV] = link
        link[self.PREV] = last
        link[self.NEXT] = self.root
        self.hits += 1
        return result

    def __setitem__(self, key, value):
        # Convert the key tensor to a tuple
        if key in self.cache:
            return
        if self.full:
            # Replace the oldest entry in the cache
            oldroot = self.root
            oldroot[self.KEY] = key
            oldroot[self.RESULT] = value
            self.root = oldroot[self.NEXT]
            oldkey = self.root[self.KEY]
            self.root[self.KEY] = self.root[self.RESULT] = None
            del self.cache[oldkey]
            self.cache[key] = oldroot
        else:
            # Store the new key-value pair in the cache
            last = self.root[self.PREV]
            link = [last, self.root, key, value]
            last[self.NEXT] = self.root[self.PREV] = self.cache[key] = link
            self.full = len(self.cache) >= self.max_size

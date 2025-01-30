import math
import time
from multiprocessing import Pool
from functools import partial
from pymatgen.analysis.structure_matcher import StructureMatcher



class UniqueStructureGetter:
    def __init__(self, **kwargs):
        """
        Initialize the UniqueStructure class with a StructureMatcher.
        """
        self.matcher_args = kwargs  # Store matcher arguments for subprocesses

    def is_similar_to_any(self, obj, unique_objects, sm):
        """
        Check if `obj` is similar to any object in `unique_objects` using StructureMatcher.
        """
        for unique_obj in unique_objects:
            if sm.fit(obj, unique_obj):
                return True
        return False

    def filter_unique_chunk(self, chunk, unique_objects, chunk_is_unique=False):
        """
        Filter unique objects from a single chunk, comparing with an existing list of unique objects.
        """
        sm = StructureMatcher(**self.matcher_args)  # Create a new matcher for each process
        chunk_unique = []
        chunk = sorted(chunk, key=len, reverse=True) # To preferentially return smallest objects

        # If chunk_is_unique is False, only process chunk sequentially (avoiding nested Pool)
        if not chunk_is_unique:
            for i, obj in enumerate(chunk):
                if not self.is_similar_to_any(obj, unique_objects + chunk[i + 1:], sm):
                    chunk_unique.append(obj)
        else:
            # Parallelize merging only when merging "unique" chunks
            with Pool() as pool:
                results = pool.map(
                    partial(self.is_similar_to_any, unique_objects=unique_objects, sm=sm), chunk
                )
                chunk_unique = [obj for obj, is_similar in zip(chunk, results) if not is_similar]

        return unique_objects + chunk_unique

    def chunk_list(self, lst, n_chunks):
        """
        Divide a list into approximately equal chunks.
        """
        chunk_size = math.ceil(len(lst) / n_chunks)
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def recursive_merge(self, chunks):
        """
        Recursively merge chunks by always merging the smallest list with the largest list.
        """
        # Sort chunks by size
        
        #chunks = sorted(chunks, key=len)

        while len(chunks) > 1:
            # Merge the first and second chunks
            first = chunks.pop(0)  
            second = chunks.pop(0)  

            # Merge the second chunk into the first chunk (chunk_is_unique=True to parallelize)
            merged = self.filter_unique_chunk(second, first, chunk_is_unique=True)

            # Re-insert the merged chunk into the sorted list
            chunks.insert(0, merged)
            print(f"Unique structures: {len(chunks[0])}")

        # The final remaining chunk is the list of unique objects
        return chunks[0]

    def filter_unique_with_recursive_chunks(self, object_list, n_chunks=4):
        """
        Main function to filter unique objects using recursive chunking and merging.
        """
        # Sort the objects by length
        object_list = sorted(object_list, key=len)
        
        # Initial chunking
        chunks = self.chunk_list(object_list, n_chunks)

        # Prepare the partial function for multiprocessing
        filter_partial = partial(self.filter_unique_chunk, unique_objects=[], chunk_is_unique=False)

        # Process each chunk independently in parallel
        with Pool() as pool:
            chunk_results = pool.map(filter_partial, chunks)
            chunk_results = sorted(chunk_results, key=lambda lst: len(min(chunk_results, key=len)))
            print(f"Unique structures: {len(chunk_results[0])}")

        # Recursively merge chunks until only one list of unique objects remains
        unique_objects = self.recursive_merge(chunk_results)
        unique_objects = sorted(unique_objects, key=len)
        print(f"Unique structures: {len(unique_objects)}")
        
        return unique_objects

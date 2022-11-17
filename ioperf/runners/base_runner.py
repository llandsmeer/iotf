__all__ = ['BaseRunner']

class BaseRunner:
    '''
    Common interface description
    '''

    def setup(self, *, ngj, ncells, argconfig):
        '''
        Set up functions, context etc to run a simulation.
        This is used instead of a constructor, so we can
        just reuse the same Runner class.

        *args and **kwargs are assumed to be in a format
        such that it can be passed unchanged to make_tf_function()
        '''
        raise NotImplementedError()

    def run_unconnected(self, nsteps, state, probe=False, **kwargs):
        '''
        Run without gap junctions (only works if setup was called
        with gap junctions disabled)
        '''
        raise NotImplementedError()

    def run_with_gap_junctions(self, nsteps, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        '''
        Run with gap junctions (only works if setup was called
        with gap junction enabled config)
        '''
        raise NotImplementedError()

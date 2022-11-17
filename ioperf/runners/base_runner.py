__all__ = ['BaseRunner']

class BaseRunner:
    '''
    Common interface description
    '''

    def tagline(self):
        '''
        This function should output a single line of data about hardware
        configuration specific to the runner
        '''
        return '<no-tagline>'

    def is_supported(self):
        '''
        Return True on supported systems, false on non-supported systems
        '''
        raise NotImplementedError()

    def setup(self, *, ngj, ncells, argconfig):
        '''
        Set up functions, context etc to run a simulation.
        This is used instead of a constructor, so we can
        just reuse the same Runner class.

        *args and **kwargs are assumed to be in a format
        such that it can be passed unchanged to make_tf_function()
        '''
        raise NotImplementedError()

    def setup_using_model_config(self, model_config, *, gap_junctions):
        assert gap_junctions in {True, False}
        if gap_junctions:
            self.setup(ncells=model_config.ncells, ngj=model_config.ngj, argconfig=dict())
        else:
            self.setup(ncells=model_config.ncells, ngj=0, argconfig=dict())

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

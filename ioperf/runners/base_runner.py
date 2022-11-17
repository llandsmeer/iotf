__all__ = ['BaseRunner']

class BaseRunner:
    '''
    Common interface description
    '''
    def setup(self, *args, **kwargs):
        raise NotImplementedError()

    def run_unconnected(self, nsteps, state, probe=False, **kwargs):
        raise NotImplementedError()

    def run_with_gap_junctions(self, nsteps, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        raise NotImplementedError()

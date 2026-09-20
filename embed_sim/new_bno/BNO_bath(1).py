from pyscf import df, ao2mo, lib
from pyscf.lib import logger
from pyscf.mp.dfmp2 import _DFINCOREERIS as _RDFINCOREERIS
from pyscf.mp.dfmp2 import _DFOUTCOREERIS as _RDFOUTCOREERIS
from pyscf.mp.dfump2 import _DFINCOREERIS as _UDFINCOREERIS
from pyscf.mp.dfump2 import _DFOUTCOREERIS as _UDFOUTCOREERIS
from functools import reduce
import numpy as np
import ctypes
import os
import itertools
import tempfile
import time
from contextlib import ExitStack


# A complete replacement of the legacy matrix construction when correct=True.
# All retained amplitudes belong to ONE EO/external local first-order state.
LOCAL_ROMP2_METHOD = 'eo-ext-wavefunction-v3-eo3-second-order-real'
_LOCAL_CACHE = 'ROMP2_EOext_rdm.npz'
_LOCAL_TEXT = ('D_MP2_core_local.txt', 'D_MP2_vir_local.txt')


def _local_real(value, name):
    a = np.asarray(value)
    if np.iscomplexobj(a):
        if a.size and np.max(np.abs(a.imag)) > 1e-12:
            raise NotImplementedError('Local ROMP2 currently requires real orbitals: ' + name)
        a = a.real
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' contains non-finite values')
    return np.asarray(a, dtype=np.float64)


def _local_reference_spaces(mf, es_mf, ao2eo, ao2core, ao2vir, log):
    """Use the determinant formed by filled core + embedded ROHF occupations.

    ONE physical Fock per spin defines all four subspace diagonalizations and
    the singles numerator. No UHF SCF, energy shifts, or interspace rotations.
    Spaces: o = EO occupied, v = EO virtual, c = core, d = D-vir.
    """
    if getattr(mf, 'with_df', None) is None:
        raise ValueError('correct=True requires a density-fitted RHF/ROHF reference')
    if np.ndim(mf.mo_coeff) != 2 or np.ndim(es_mf.mo_coeff) != 2:
        raise TypeError('correct=True requires spatial RHF/ROHF reference orbitals')
    if mf.mol.spin < 0:
        raise NotImplementedError('The derivation assumes singly occupied alpha orbitals')
    s = _local_real(mf.get_ovlp(), 'AO overlap')
    eo, core, dvir = [_local_real(x, name) for x, name in
                     ((ao2eo, 'EO'), (ao2core, 'core'), (ao2vir, 'D-vir'))]
    basis = np.hstack((eo, core, dvir))
    if basis.shape != s.shape or not np.allclose(basis.T @ s @ basis,
                                                np.eye(s.shape[0]), atol=1e-7, rtol=0):
        raise ValueError('EO/core/D-vir must form a complete mutually S-orthonormal basis')
    mo = _local_real(es_mf.mo_coeff, 'embedded MOs')
    occ = _local_real(es_mf.mo_occ, 'embedded occupations')
    if mo.shape != (eo.shape[1], eo.shape[1]) or occ.shape != (eo.shape[1],):
        raise ValueError('Embedded MO dimensions do not match the supplied EO space')
    if not np.all(np.min(np.abs(occ[:, None] - [0., 1., 2.]), axis=1) < 1e-7):
        raise ValueError('Embedded ROHF occupations must be 0, 1, or 2')
    embedded = eo @ mo
    if not np.allclose(embedded.T @ s @ embedded, np.eye(len(occ)), atol=1e-7, rtol=0):
        raise ValueError('Embedded ROHF orbitals are not S-orthonormal')
    initial = []
    for mask in (occ > .5, occ > 1.5):
        initial.append({'o': embedded[:, mask], 'v': embedded[:, ~mask],
                        'c': core, 'd': dvir})
    counts = tuple(x['o'].shape[1] + core.shape[1] for x in initial)
    if counts != tuple(mf.mol.nelec):
        raise ValueError('Core + EO occupations do not reproduce the molecular spin populations')
    dm = np.asarray([x['o'] @ x['o'].T + core @ core.T for x in initial])
    vj, vk = mf.get_jk(dm=dm)
    fock = _local_real(mf.get_hcore() + vj[0] + vj[1] - vk, 'reference spin Fock')
    if fock.shape != (2,) + s.shape:
        raise ValueError('Expected two collinear spin Fock matrices')
    original_dm = np.asarray(mf.make_rdm1())
    if original_dm.ndim == 2:
        original_dm = np.stack((original_dm * .5, original_dm * .5))
    reference_delta = float(np.linalg.norm(dm - original_dm))
    log.info('Local ROMP2 reference: filled core + embedded ROHF; AO density change %.3e',
             reference_delta)
    spaces = []
    for spin, groups in enumerate(initial):
        result = {}
        for key, c in groups.items():
            projected = c.T @ fock[spin] @ c
            energy, u = np.linalg.eigh((projected + projected.T) * .5)
            result[key] = (c @ u, energy, u)
        spaces.append(result)
        log.info('Local ROMP2 %s dimensions: EO occ %d, EO vir %d, core %d, D-vir %d',
                 ('alpha', 'beta')[spin], *(result[k][0].shape[1] for k in ('o','v','c','d')))
    return spaces, fock, reference_delta


class _LocalDFPairs:
    """At most eight occupied/virtual DF pairs, with selective disk reads."""
    def __init__(self, mf, spaces, log, stack):
        self.mf, self.spaces, self.log, self.stack = mf, spaces, log, stack
        self.pairs = {}

    def ensure(self, spin, occupied, virtual):
        key = (spin, occupied, virtual)
        if key not in self.pairs:
            left, right = (self.spaces[spin][k][0] for k in (occupied, virtual))
            naux = self.mf.with_df.get_naoaux()
            need = left.shape[1] * right.shape[1] * naux * 8 / 1e6
            available = max(0., self.mf.max_memory - lib.current_memory()[0])
            outcore = need > .2 * available
            cls = _RDFOUTCOREERIS if outcore else _RDFINCOREERIS
            self.log.info('Local ROMP2 DF pair %s %s-%s: (%d,%d,%d), %s',
                          ('alpha','beta')[spin], occupied, virtual,
                          left.shape[1], right.shape[1], naux,
                          'disk' if outcore else 'memory')
            self.log.stdout.flush()
            eris = cls(self.mf.with_df, left, right, self.mf.max_memory,
                       verbose=self.log.verbose, stdout=self.log.stdout)
            try:
                eris.build()
            except BaseException:
                if getattr(eris, 'feri', None) is not None:
                    eris.feri.close()
                raise
            if getattr(eris, 'feri', None) is not None:
                self.stack.callback(eris.feri.close)
            self.pairs[key] = eris
        return self.pairs[key]

    def read(self, spin, occupied, virtual, irange, arange):
        eris = self.ensure(spin, occupied, virtual)
        i0, i1 = irange
        a0, a1 = arange
        nv, nl = eris.nvir, eris.naux
        if isinstance(eris.ovL, np.ndarray):
            return eris.ovL.reshape(eris.nocc, nv, nl)[i0:i1, a0:a1]
        if a0 == 0 and a1 == nv:
            return eris.get_occ_blk(i0, i1)
        out = np.empty((i1-i0, a1-a0, nl))
        for row, i in enumerate(range(i0, i1)):
            out[row] = eris.ovL[i*nv+a0:i*nv+a1]
        return out


def _local_keep_double(groups):
    """V3: retain doubles with exactly three EO (tilde) indices, in all blocks.

    One-EO and two-EO doubles are omitted before integral contractions.
    This rule applies only to doubles; cross-boundary singles are retained.
    """
    return sum(g in ('o','v') for g in groups) == 3


def _local_terms(target, spin):
    """V3: one same-spin + one opposite-spin TT term in each density block.

    (groups I,J,A,B; spins I,J,A,B; coefficient). All unrestricted sums are
    over complete spin/subspace blocks. The external free index is the only
    non-EO index; identical-space same-spin pairs retain coefficient 1/2.
    """
    same, opp = (spin,)*4, (spin, 1-spin, spin, 1-spin)
    if target == 'core':
        groups = ('c','o','v','v')
    else:
        groups = ('o','o','d','v')
    return [(groups, same, .5), (groups, opp, 1.)]


def _local_tile_shape(shape, free_axis, naux, same_spin, budget):
    dims = list(shape)
    def memory(x):
        i,j,a,b = x
        factors = i*a + j*b + (i*b+j*a if same_spin else 0)
        return 48 * np.prod(x, dtype=np.int64) + 8*naux*factors + 8*x[free_axis]**2
    while memory(dims) > budget:
        movable = [k for k in range(4) if k != free_axis and dims[k] > 1]
        if not movable:
            raise MemoryError('Insufficient memory for a local ROMP2 contraction tile; '
                              'increase mf.max_memory')
        k = max(movable, key=lambda k: dims[k])
        dims[k] = (dims[k] + 1) // 2
    return tuple(dims)


def _local_amplitude_tile(pairs, spaces, groups, spins, ranges):
    if not _local_keep_double(groups):
        return np.zeros(tuple(hi-lo for lo,hi in ranges))
    i,j,a,b = groups
    si,sj,sa,sb = spins
    ir,jr,ar,br = ranges
    # I/A and J/B have matching spin by construction. The other ordering of
    # opposite-spin particles is accounted for by antisymmetry and the sums.
    ia = pairs.read(si, i, a, ir, ar)
    jb = pairs.read(sj, j, b, jr, br)
    t = lib.einsum('iaL,jbL->ijab', ia, jb)
    del ia, jb
    if si == sj:
        ib = pairs.read(si, i, b, ir, br)
        ja = pairs.read(sj, j, a, jr, ar)
        t -= lib.einsum('ibL,jaL->ijab', ib, ja)
        del ib, ja
    e = [spaces[s][g][1][lo:hi] for g,s,(lo,hi) in zip(groups,spins,ranges)]
    denominator = (e[0][:,None,None,None]+e[1][None,:,None,None]
                   -e[2][None,None,:,None]-e[3][None,None,None,:])
    # Repeated same-spin indices are Pauli-forbidden determinants, not poles.
    if si == sj and i == j:
        repeated = (np.arange(*ir)[:,None] == np.arange(*jr)[None,:])[:,:,None,None]
        np.copyto(t, 0., where=repeated)
        np.copyto(denominator, 1., where=repeated)
    if sa == sb and a == b:
        repeated = (np.arange(*ar)[:,None] == np.arange(*br)[None,:])[None,None,:,:]
        np.copyto(t, 0., where=repeated)
        np.copyto(denominator, 1., where=repeated)
    if np.any(np.abs(denominator) < 1e-12):
        raise FloatingPointError('Near-zero local ROMP2 denominator: %s %s' % (groups,spins))
    t /= denominator
    if not np.all(np.isfinite(t)):
        raise FloatingPointError('Non-finite local ROMP2 amplitudes')
    return t


def _local_contract_term(mf, spaces, pairs, groups, spins, coefficient, free_axis, log):
    shape = tuple(spaces[s][g][0].shape[1] for g,s in zip(groups,spins))
    nfree = shape[free_axis]
    gram = np.zeros((nfree,nfree))
    if 0 in shape or not _local_keep_double(groups):
        return gram, 0.
    si,sj,sa,sb = spins
    i,j,a,b = groups
    pairs.ensure(si,i,a)
    pairs.ensure(sj,j,b)
    if si == sj:
        pairs.ensure(si,i,b)
        pairs.ensure(sj,j,a)
    naux = mf.with_df.get_naoaux()
    available = max(0., mf.max_memory-lib.current_memory()[0]) * 1e6
    tile = _local_tile_shape(shape, free_axis, naux, si==sj, min(.6*available, 512*1024**2))
    ranges = [list(lib.prange(0,n,k)) for n,k in zip(shape,tile)]
    total = int(np.prod([len(r) for r in ranges]))
    log.info('  groups %s spins %s coefficient %.1f; tile %s; %d tiles',
             ''.join(groups), spins, coefficient, tile, total)
    log.stdout.flush()
    last = start = time.monotonic()
    for done, slices in enumerate(itertools.product(*ranges), 1):
        t = _local_amplitude_tile(pairs, spaces, groups, spins, slices)
        x = np.ascontiguousarray(np.moveaxis(t, free_axis, 0)).reshape(nfree,-1)
        gram += coefficient * (x @ x.T)
        del x, t
        now = time.monotonic()
        if done == 1 or done == total or now-last >= 30:
            log.info('    tiles %d/%d; elapsed %.1f s', done,total,now-start)
            log.stdout.flush()
            last = now
    # Summing this over every external block counts each determinant once:
    # traces count its external holes + particles, so divide by that number.
    nexternal = sum(g in ('c','d') for g in groups)
    return gram, float(np.trace(gram)) / nexternal


def _local_contract_density(mf, spaces, fock, log):
    data = {}
    norm_singles = norm_doubles = 0.
    with ExitStack() as stack:
        pairs = _LocalDFPairs(mf, spaces, log, stack)
        for spin, spin_name in enumerate(('alpha','beta')):
            for target, key, free_axis in (('core','c',0),('vir','d',2)):
                n = spaces[spin][key][0].shape[1]
                weight = np.zeros((n,n))
                if n:
                    terms = _local_terms(target,spin)
                    for number, (groups,spins,coef) in enumerate(terms,1):
                        log.info('Local ROMP2 %s %s: doubles term %d/%d',
                                 target,spin_name,number,len(terms))
                        part, norm = _local_contract_term(mf,spaces,pairs,groups,spins,
                                                          coef,free_axis,log)
                        weight += part
                        norm_doubles += norm
                        del part
                    occupied, virtual = ('c','v') if target=='core' else ('o','d')
                    co,eo,_ = spaces[spin][occupied]
                    cv,ev,_ = spaces[spin][virtual]
                    denominator = eo[:,None]-ev[None,:]
                    if np.any(np.abs(denominator)<1e-12):
                        raise FloatingPointError('Near-zero local ROMP2 singles denominator')
                    t1 = (co.T @ fock[spin] @ cv) / denominator
                    if not np.all(np.isfinite(t1)):
                        raise FloatingPointError('Non-finite local ROMP2 singles amplitudes')
                    part = t1 @ t1.T if target=='core' else t1.T @ t1
                    weight += part
                    norm_singles += float(np.trace(part))
                weight = (weight+weight.T)*.5
                data[target+'_'+spin_name] = np.eye(n)-weight if target=='core' else weight
                data['rotation_'+target+'_'+spin_name] = spaces[spin][key][2]
    for target in ('core','vir'):
        data[target] = sum(data['rotation_'+target+'_'+s] @ data[target+'_'+s]
                           @ data['rotation_'+target+'_'+s].T for s in ('alpha','beta'))
        data[target] = (data[target]+data[target].T)*.5
    data['norm1_singles'] = np.asarray(norm_singles)
    data['norm1_doubles'] = np.asarray(norm_doubles)
    data['norm1'] = np.asarray(norm_singles+norm_doubles)
    data['normalization_sq'] = np.asarray(1./(1.+norm_singles+norm_doubles))
    log.info('Local ROMP2 ||delta Psi1||^2 = %.12g (singles %.12g, doubles %.12g)',
             norm_singles+norm_doubles,norm_singles,norm_doubles)
    log.info('Selection uses SECOND-ORDER RDMs; exact normalized-state M^2 = %.12g',
             float(data['normalization_sq']))
    log.info('External traces: core holes %.12g, D-vir particles %.12g '
             '(need not be equal; EO carries the difference)',
             2*data['core'].shape[0]-np.trace(data['core']),np.trace(data['vir']))
    return data


def get_local_ROMP2_density(mf, es_mf, ao2eo, ao2core, ao2vir, readmp2=False, verbose=None):
    """V3: construct four second-order blocks with exactly-three-EO doubles.

    One-EO and two-EO doubles are omitted. The four cross-boundary singles
    classes are unchanged. With readmp2=True, an existing NPZ cache is read
    directly, without constructing the reference or comparing its metadata.
    Set readmp2=False to recompute V3 matrices, including when replacing a
    V1/V2 cache. Missing caches are computed; unreadable caches raise an error.

    Returns common-basis core/vir matrices, four semicanonical spin blocks,
    their rotations, and the first-order norm. No legacy matrix is read/added.
    For the EXACT normalized reference+first-order state use I-M^2*(I-Doo)
    and M^2*Dvv, with M^2=data['normalization_sq']; that is not the default
    second-order ROMP2 selection criterion requested in the four explicit sums.
    """
    log = logger.new_logger(mf,verbose)
    if readmp2 and os.path.isfile(_LOCAL_CACHE):
        with np.load(_LOCAL_CACHE,allow_pickle=False) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        log.info('Read local ROMP2 RDM cache directly: %s', _LOCAL_CACHE)
        if 'method' in data:
            log.info('Cached RDM method: %s', str(data['method']))
    else:
        if readmp2:
            log.info('Local ROMP2 RDM cache not found; computing V3 matrices')
        log.info('Local ROMP2 V3 doubles: retain exactly 3 EO indices; '
                 'omit 1-EO and 2-EO doubles; cross-boundary singles retained')
        spaces,fock,reference_delta = _local_reference_spaces(mf,es_mf,ao2eo,ao2core,ao2vir,log)
        data = _local_contract_density(mf,spaces,fock,log)
        data.update(method=np.asarray(LOCAL_ROMP2_METHOD),
                    reference_density_change=np.asarray(reference_delta))
        path = None
        try:
            with tempfile.NamedTemporaryFile(dir='.',prefix='local_romp2_',suffix='.npz',
                                             delete=False) as stream:
                path = stream.name
                np.savez(stream,**data)
            os.replace(path,_LOCAL_CACHE)
            path = None
        finally:
            if path is not None and os.path.exists(path):
                os.remove(path)
    for path,key in zip(_LOCAL_TEXT,('core','vir')):
        np.savetxt(path,data[key],fmt='%.16e')
    return data


def _local_select_bath(data, lo2core, lo2vir, eta, ao, log):
    if not np.isscalar(eta) or not np.isfinite(eta) or eta < 0:
        raise ValueError('eta must be a nonnegative threshold or positive integer bath count')
    dc,dv = data['core'],data['vir']
    if dc.shape!=(lo2core.shape[1],)*2 or dv.shape!=(lo2vir.shape[1],)*2:
        raise ValueError('Local ROMP2 density dimensions do not match the environment bases')
    ec,uc = np.linalg.eigh(dc)
    ev,uv = np.linalg.eigh(dv)
    if eta >= 1:
        if float(eta) != int(eta):
            raise ValueError('eta >= 1 denotes an integer number of added bath orbitals')
        eta = choose_eta_for_nbath(ec,ev,int(eta),tol=1e-12,strict=True)
        log.info('Local ROMP2 threshold for requested bath count: %.16g',eta)
    mc = (ec < 2-eta) | (ec > 2+eta) if ao else ec < 2-eta
    mv = ev > eta
    bc,bv = lo2core @ uc[:,mc],lo2vir @ uv[:,mv]
    log.info('Unified local ROMP2 bath: %d added (%d core, %d D-vir)',
             int(mc.sum()+mv.sum()),int(mc.sum()),int(mv.sum()))
    return np.hstack((bc,bv)),lo2core @ uc[:,~mc],lo2vir @ uv[:,~mv]


def _load_mp2_matrix_allow_empty(fname):
    """
    返回:
      - None: 文件不存在 / 只有空白 / 读取失败 / 形状非法
      - (n,n) ndarray: 合法方阵
    """
    if (not os.path.isfile(fname)) or (os.path.getsize(fname) == 0):
        return None
    try:
        M = np.loadtxt(fname)
    except Exception:
        return None
    if M.ndim != 2 or (M.shape[0] != M.shape[1]):
        return None
    return M

def choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True):
    """
    选择 eta 使得：
      nbath_new = sum(eigvals_core < 2-eta) + sum(eigvals_vir > eta) == G

    参数
    ----
    eigvals_core, eigvals_vir : 1D arrays
    G : int, 目标 bath 个数
    tol : 数值容差
    strict : True 时用严格不等号（匹配你原代码）；False 时改为带容差的比较

    返回
    ----
    eta : float
    """
    eigvals_core = np.asarray(eigvals_core).ravel()
    eigvals_vir  = np.asarray(eigvals_vir ).ravel()

    # 把两个条件都写成 “eta < threshold”
    # core: eta < (2 - eigvals_core)
    # vir : eta < (eigvals_vir)
    T = np.concatenate((2.0 - eigvals_core, eigvals_vir))

    # 可达到的 bath 数范围：eta -> +inf => 0； eta -> -inf => T.size
    K = T.size
    if not (0 <= G <= K):
        raise ValueError(f"G must be between 0 and {K}, got {G}")

    # 用 unique 阈值把实数轴分段；在每个相邻阈值之间计数不变
    u = np.unique(T)
    u.sort()          # 升序
    u = u[::-1]       # 降序

    # 候选 eta：两两相邻阈值的中点（保证不踩到等号）
    # shape: (len(u)-1,)
    if u.size == 1:
        # 所有阈值相同，只能得到 K 或 0（取决于 eta < u[0] 与否）
        if G == K:
            return u[0] - 10*tol
        if G == 0:
            return u[0] + 10*tol
        raise ValueError("All thresholds equal -> cannot realize intermediate G.")

    mids = 0.5 * (u[:-1] + u[1:])  # 在这些点上 nbath 恒定

    # 计算每个 mid 对应的 nbath（向量化，无循环）
    if strict:
        counts = np.sum(T[None, :] > mids[:, None], axis=1)
    else:
        counts = np.sum(T[None, :] > (mids[:, None] + tol), axis=1)

    # 还要考虑两端：
    # eta > max(T) -> 0
    # eta < min(T) -> K
    # 我们把两端也并入候选集合
    eta_hi = u[0] + 10*tol   # 使 eta > max(T)
    eta_lo = u[-1] - 10*tol  # 使 eta < min(T)

    # 查找是否存在 counts==G 的 mid
    hit = np.where(counts == G)[0]
    if hit.size > 0:
        return float(mids[hit[0]])

    # 两端是否命中
    if G == 0:
        return float(eta_hi)
    if G == K:
        return float(eta_lo)

    # 否则说明由于阈值重复/跳变，无法精确得到 G
    # 这里给出“最接近”的 eta（也可改成直接 raise）
    idx = np.argmin(np.abs(counts - G))
    raise ValueError(f"Cannot achieve nbath_new == {G} exactly. Closest is {counts[idx]} at eta={mids[idx]}.")


def get_RMP2_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir, eta=1e-4, verbose=None):
    
    log = logger.new_logger(mf, verbose=verbose)
    log.info('')
    log.info('constructing RMP2 bath')
    
    nocc = (mf.mo_occ>0).sum()
    nvir = (mf.mo_occ==0).sum()
    
    occ_coeff = mf.mo_coeff[:,mf.mo_occ>0]
    vir_coeff = mf.mo_coeff[:,mf.mo_occ==0]
    es_occ_coeff = lib.dot(ao2eo, es_mf.mo_coeff[:,es_mf.mo_occ>0])
    es_vir_coeff = lib.dot(ao2eo, es_mf.mo_coeff[:,es_mf.mo_occ==0])
    
    occ_energy = mf.mo_energy[mf.mo_occ>0]
    vir_energy = mf.mo_energy[mf.mo_occ==0]
    es_occ_energy = es_mf.mo_energy[es_mf.mo_occ>0]
    es_vir_energy = es_mf.mo_energy[es_mf.mo_occ==0]
    
    def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
        log = logger.new_logger(mf, verbose)
    
        with_df = getattr(mf, 'with_df', None)
        assert( with_df is not None )
    
        if with_df._cderi is None:
            log.debug('Caching ovL-type integrals directly')
            if with_df.auxmol is None:
                with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
        else:
            log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
    
        assert (occ_coeff is not None and vir_coeff is not None)
    
        # determine incore or outcore
        nocc = occ_coeff.shape[1]
        nvir = vir_coeff.shape[1]
        naux = with_df.get_naoaux()
    
        if ovL is not None:
            if isinstance(ovL, np.ndarray):
                outcore = False
            elif isinstance(ovL, str):
                outcore = True
            else:
                log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                          type(ovL))
                raise TypeError
        else:
            mem_now = mf.max_memory - lib.current_memory()[0]
            mem_df = nocc*nvir*naux*8/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            
            outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
        log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')
    
        if outcore:
            eris = _RDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                  ovL=ovL, ovL_to_save=ovL_to_save,
                                  verbose=log.verbose, stdout=log.stdout)
        else:
            eris = _RDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                 ovL=ovL,
                                 verbose=log.verbose, stdout=log.stdout)
        eris.build()
    
        return eris
    
    def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
    
        log = logger.new_logger(mf, verbose)
    
        assert (ao2mo is not None)
    
        nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
        assert (occ_energy is not None and vir_energy is not None)
        moevv = np.asarray(vir_energy[:,None] + vir_energy, order='C')
    
        mem_avail = mf.max_memory - lib.current_memory()[0]
    
        if with_t2:
            t2 = np.zeros((nocc,nocc,nvir,nvir), dtype=eris.dtype)
            t2_ptr = t2.ctypes.data_as(ctypes.c_void_p)
            mem_avail -= t2.size * eris.dsize / 1e6
        else:
            t2 = None
            t2_ptr = lib.c_null_ptr()
    
        if mem_avail < 0:
            log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
            raise MemoryError
    
        libmp = lib.load_library('libmp')
        drv = libmp.MP2_contract_d
    
        # determine occ blksize
        if isinstance(eris.ovL, np.ndarray):    # incore ovL
            occ_blksize = nocc
        else:   # outcore ovL
            # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
            occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvir**2)/(2*naux*nvir)))
            occ_blksize = min(nocc, max(1, occ_blksize))
    
        log.debug('occ blksize for %s loop: %d/%d', mf.__class__.__name__, occ_blksize, nocc)
    
        cput1 = (logger.process_clock(), logger.perf_counter())
    
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
            nocci = i1-i0
            iaL = eris.get_occ_blk(i0,i1)
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
                noccj = j1-j0
                if ibatch == jbatch:
                    jbL = iaL
                else:
                    jbL = eris.get_occ_blk(j0,j1)
    
                ed = np.zeros(1, dtype=np.float64)
                ex = np.zeros(1, dtype=np.float64)
                moeoo_block = np.asarray(
                    occ_energy[i0:i1,None] + occ_energy[j0:j1], order='C')
                s2symm = 1
                t2_ex = 0
                drv(
                    ed.ctypes.data_as(ctypes.c_void_p),
                    ex.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(s2symm),
                    iaL.ctypes.data_as(ctypes.c_void_p),
                    jbL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(i0), ctypes.c_int(j0),
                    ctypes.c_int(nocci), ctypes.c_int(noccj),
                    ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
                    moeoo_block.ctypes.data_as(ctypes.c_void_p),
                    moevv.ctypes.data_as(ctypes.c_void_p),
                    t2_ptr, ctypes.c_int(t2_ex)
                )
    
                jbL = None
            iaL = None
    
            cput1 = log.timer_debug1('i-block [%d:%d]/%d' % (i0,i1,nocc), *cput1)
    
        return t2
    
    def _gamma1_intermediates(mf, t2=None, eris=None):
        assert (t2 is not None)
        nocc, nocc, nvir, nvir = t2.shape
        dtype = t2.dtype
    
        dm1occ = np.zeros((nocc,nocc), dtype=dtype)
        dm1vir = np.zeros((nvir,nvir), dtype=dtype)
        for i in range(nocc):
            t2i = t2[i]
            l2i = t2i
            dm1vir += lib.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                    - lib.einsum('jca,jbc->ba', l2i, t2i)
            dm1occ += lib.einsum('iab,jab->ij', l2i, t2i) * 2 \
                    - lib.einsum('iab,jba->ij', l2i, t2i)
        
        dm1occ *= -1
        dm1vir += dm1vir.T
        dm1occ += dm1occ.T
        dm1occ[np.diag_indices(nocc)] += 2
        return dm1occ, dm1vir
    
    eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff)
    eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff)
    
    t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov)
    t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV)
    
    D_IJ = _gamma1_intermediates(mf, t_IJab, eris_Ov)[0]
    D_AB = _gamma1_intermediates(mf, t_ijAB, eris_oV)[1]
    
    S = mf.get_ovlp()
    D_IJ_ao = lib.einsum('pi,ij,qj->pq', occ_coeff, D_IJ, occ_coeff)
    D_AB_ao = lib.einsum('pi,ij,qj->pq', vir_coeff, D_AB, vir_coeff)
    D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
    D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))
    
    bins = np.array([10**-x for x in range(0,11)][::-1])
    eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
    histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
    log.info('Occupied BNO histogram')
    log.info('%s',histogram_core)
    log.info('')
    
    eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
    histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
    log.info('Virtual BNO histogram')
    log.info('%s',histogram_vir)
    log.info('')
    
    MP2_bath_core = (eigvals_core < 2 - eta)
    MP2_bath_vir = (eigvals_vir > eta)

    lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
    lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
    lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
    lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
    lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])
    
    nbath_new_core = MP2_bath_core.sum()
    nbath_new_vir = MP2_bath_vir.sum()
    nbath_new = nbath_new_core + nbath_new_vir
    ncore_new = (~MP2_bath_core).sum()
    log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
    # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
    log.info('')
    
    return lo2MP2_bath, lo2MP2_core, lo2MP2_vir

def get_UMP2_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir, eta=1e-2, verbose=None):
    
    log = logger.new_logger(mf, verbose=verbose)
    log.info('')
    log.info('constructing UMP2 bath')
    
    mf = mf.to_uhf()
    es_mf = es_mf.to_uhf()
    
    nocc = [(mf.mo_occ[i]>0).sum() for i in range(2)]
    nvir = [(mf.mo_occ[i]==0).sum() for i in range(2)]
    
    occ_coeff = [mf.mo_coeff[i][:,mf.mo_occ[i]>0] for i in range(2)]
    vir_coeff = [mf.mo_coeff[i][:,mf.mo_occ[i]==0] for i in range(2)]
    es_occ_coeff = [lib.dot(ao2eo, es_mf.mo_coeff[i][:,es_mf.mo_occ[i]>0]) for i in range(2)]
    es_vir_coeff = [lib.dot(ao2eo, es_mf.mo_coeff[i][:,es_mf.mo_occ[i]==0]) for i in range(2)]
    
    occ_energy = [mf.mo_energy[i][mf.mo_occ[i]>0] for i in range(2)]
    vir_energy = [mf.mo_energy[i][mf.mo_occ[i]==0] for i in range(2)]
    es_occ_energy = [es_mf.mo_energy[i][es_mf.mo_occ[i]>0] for i in range(2)]
    es_vir_energy = [es_mf.mo_energy[i][es_mf.mo_occ[i]==0] for i in range(2)]
    
    def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
        log = logger.new_logger(mf, verbose)
    
        with_df = getattr(mf, 'with_df', None)
        assert( with_df is not None )
    
        if with_df._cderi is None:
            log.debug('Caching ovL-type integrals directly')
            if with_df.auxmol is None:
                with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
        else:
            log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
    
        assert (occ_coeff is not None and vir_coeff is not None)
    
        # determine incore or outcore
        nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
        nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
        naux = with_df.get_naoaux()
    
        if ovL is not None:
            if isinstance(ovL, np.ndarray):
                outcore = False
            elif isinstance(ovL, str):
                outcore = True
            else:
                log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                          type(ovL))
                raise TypeError
        else:
            mem_now = mf.max_memory - lib.current_memory()[0]
            mem_df = sum(nocc*nvir)*8/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            
            outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
        log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')
    
        if outcore:
            eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                  ovL=ovL, ovL_to_save=ovL_to_save,
                                  verbose=log.verbose, stdout=log.stdout)
        else:
            eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                 ovL=ovL,
                                 verbose=log.verbose, stdout=log.stdout)
        eris.build()
    
        return eris
    
    def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
    
        log = logger.new_logger(mf, verbose)
    
        assert (ao2mo is not None)
    
        nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
        nvirmax = max(nvir)
        assert (occ_energy is not None and vir_energy is not None)
    
        mem_avail = mf.max_memory - lib.current_memory()[0]
    
        if with_t2:
            t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                  np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                  np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
            t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
            mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
        else:
            t2 = None
            t2_ptr = [lib.c_null_ptr()] * 3
    
        if mem_avail < 0:
            log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
            raise MemoryError
    
        libmp = lib.load_library('libmp')
        drv = libmp.MP2_contract_d
    
        # determine occ blksize
        if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
            occ_blksize = nocc
        else:   # outcore ovL
            # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
            occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
            occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]
    
        log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                  occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])
    
        cput1 = (logger.process_clock(), logger.perf_counter())
    
        for s in [0,1]:
            s_t2 = 0 if s == 0 else 2
            moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
            for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                nocci = i1-i0
                iaL = eris.get_occ_blk(s,i0,i1)
                for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                    noccj = j1-j0
                    if ibatch == jbatch:
                        jbL = iaL
                    else:
                        jbL = eris.get_occ_blk(s,j0,j1)
    
                    ed = np.zeros(1, dtype=np.float64)
                    ex = np.zeros(1, dtype=np.float64)
                    moeoo_block = np.asarray(
                        occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                    s2symm = 1
                    t2_ex = True
                    drv(
                        ed.ctypes.data_as(ctypes.c_void_p),
                        ex.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(s2symm),
                        iaL.ctypes.data_as(ctypes.c_void_p),
                        jbL.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(i0), ctypes.c_int(j0),
                        ctypes.c_int(nocci), ctypes.c_int(noccj),
                        ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                        moeoo_block.ctypes.data_as(ctypes.c_void_p),
                        moevv.ctypes.data_as(ctypes.c_void_p),
                        t2_ptr[s_t2], ctypes.c_int(t2_ex)
                    )
    
                    jbL = None
                iaL = None
    
                cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                         *cput1)
                
        # opposite spin
        sa, sb = 0, 1
        drv = libmp.MP2_OS_contract_d
        moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
            nocci = i1-i0
            iaL = eris.get_occ_blk(sa,i0,i1)
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                noccj = j1-j0
                jbL = eris.get_occ_blk(sb,j0,j1)
    
                ed = np.zeros(1, dtype=np.float64)
                moeoo_block = np.asarray(
                    occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                drv(
                    ed.ctypes.data_as(ctypes.c_void_p),
                    iaL.ctypes.data_as(ctypes.c_void_p),
                    jbL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(i0), ctypes.c_int(j0),
                    ctypes.c_int(nocci), ctypes.c_int(noccj),
                    ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                    ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                    ctypes.c_int(naux),
                    moeoo_block.ctypes.data_as(ctypes.c_void_p),
                    moevv.ctypes.data_as(ctypes.c_void_p),
                    t2_ptr[1]
                )
    
                jbL = None
            iaL = None
    
            cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                     *cput1)
    
        return t2
    
    def _gamma1_intermediates(mf, t2=None, eris=None):
        assert (t2 is not None)
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2[1].shape
        
        dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
        dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
        doob  = lib.einsum('imef,jmef->ij', t2bb, t2bb) *-.5
        doob -= lib.einsum('mief,mjef->ij', t2ab, t2ab)
    
        dvva  = lib.einsum('mnae,mnbe->ba', t2aa, t2aa) * .5
        dvva += lib.einsum('mnae,mnbe->ba', t2ab, t2ab)
        dvvb  = lib.einsum('mnae,mnbe->ba', t2bb, t2bb) * .5
        dvvb += lib.einsum('mnea,mneb->ba', t2ab, t2ab)
        
        dooa += dooa.T
        doob += doob.T
        dvva += dvva.T
        dvvb += dvvb.T
        dooa *= 0.5
        doob *= 0.5
        dvva *= 0.5
        dvvb *= 0.5
        dooa[np.diag_indices(nocca)] += 1
        doob[np.diag_indices(noccb)] += 1
        
        dm1occ = [dooa,doob]
        dm1vir = [dvva,dvvb]
        return dm1occ, dm1vir
    
    eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
    eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)
    
    t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose)
    t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose)
    
    D_IJ = _gamma1_intermediates(mf, t_IJab, eris_Ov)[0]
    D_AB = _gamma1_intermediates(mf, t_ijAB, eris_oV)[1]
    
    S = mf.get_ovlp()
    D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
    D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])
    D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
    D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))
    
    bins = np.array([10**-x for x in range(0,11)][::-1])
    eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
    histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
    log.info('Occupied BNO histogram')
    log.info('%s',histogram_core)
    log.info('')
    
    eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
    histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
    log.info('Virtual BNO histogram')
    log.info('%s',histogram_vir)
    log.info('')
    
    MP2_bath_core = (eigvals_core < 2 - eta)
    MP2_bath_vir = (eigvals_vir > eta)
    lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
    lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
    lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
    lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
    lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])
    
    nbath_new_core = MP2_bath_core.sum()
    nbath_new_vir = MP2_bath_vir.sum()
    nbath_new = nbath_new_core + nbath_new_vir
    ncore_new = (~MP2_bath_core).sum()
    log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
    # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
    log.info('')
    
    return lo2MP2_bath, lo2MP2_core, lo2MP2_vir

def get_ROMP2_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir, ao = False, readmp2 = False, eta=1e-2, verbose=None, correct=False):
    """correct=False: original algorithm; True: unified local-wavefunction RDMs."""
    if not isinstance(correct, (bool, np.bool_)):
        raise TypeError('correct must be True or False')
    if correct:
        log = logger.new_logger(mf, verbose)
        log.info('ROMP2 bath method: %s', LOCAL_ROMP2_METHOD)
        data = get_local_ROMP2_density(mf, es_mf, ao2eo, ao2core, ao2vir,
                                       readmp2=readmp2, verbose=verbose)
        return _local_select_bath(data, lo2core, lo2vir, eta, ao, log)
    log = logger.new_logger(mf, verbose=verbose)
    if readmp2:
        log.info('========== Read from MP2.TXT files ===========')
        if eta < 1:
            log.info('')
            log.info('========== eta < 1 ===========')
            log.info('constructing new ROMP2 bath')

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            S = mf.get_ovlp()
            D_MP2_core = _load_mp2_matrix_allow_empty('D_MP2_core.txt')
            D_MP2_vir  = _load_mp2_matrix_allow_empty('D_MP2_vir.txt')

            if D_MP2_core is None:
                log.info('D_MP2_core.txt is empty/unavailable -> no bath added from core.')
                eigvals_core = np.array([], dtype=float)
                eigvecs_core = None
                MP2_bath_core = np.zeros(lo2core.shape[1], dtype=bool)  # 全 False
            else:
                eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
                if ao:
                    log.info('Bath expansion with AO-DMET')
                    MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
                else:
                    MP2_bath_core = (eigvals_core < 2 - eta)

            bins = np.array([10**-x for x in range(0,11)][::-1])
            histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
            log.info('Occupied BNO histogram')
            log.info('%s',histogram_core)
            log.info('')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            MP2_bath_vir = (eigvals_vir > eta)

            if MP2_bath_core.size == 0:
                lo2MP2_bath_core = lo2core[:, :0]
                lo2MP2_core = lo2core
            else:
                lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
                lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])

            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum() if MP2_bath_core.size else 0
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum() if MP2_bath_core.size else lo2core.shape[1]
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')

        else:
            G = eta
            log.info('')
            log.info('========== eta >= 1 ===========')
            log.info('constructing new ROMP2 bath')

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            S = mf.get_ovlp()
            D_MP2_core = _load_mp2_matrix_allow_empty('D_MP2_core.txt')
            D_MP2_vir  = _load_mp2_matrix_allow_empty('D_MP2_vir.txt')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)

            bins = np.array([10**-x for x in range(0,11)][::-1])
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            if D_MP2_core is None:
                log.info('D_MP2_core.txt is empty/unavailable -> no bath added from core.')
                eigvals_core = np.array([], dtype=float)
                eigvecs_core = None
                MP2_bath_core = np.zeros(lo2core.shape[1], dtype=bool)  # 全 False
                eta = choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True)
                log.info("================This is new eta================")
                log.info(str(eta))
            else:
                eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
                eta = choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True)
                log.info("================This is new eta================")
                log.info(str(eta))
                if ao:
                    log.info('Bath expansion with AO-DMET')
                    MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
                else:
                    MP2_bath_core = (eigvals_core < 2 - eta)
            
            if D_MP2_core is not None:
                bins = np.array([10**-x for x in range(0,11)][::-1])
                histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
                log.info('Occupied BNO histogram')
                log.info('%s', histogram_core)
                log.info('')
            else:
                log.info('Occupied BNO histogram skipped (empty core MP2 matrix).')

            MP2_bath_vir = (eigvals_vir > eta)

            if MP2_bath_core.size == 0:
                lo2MP2_bath_core = lo2core[:, :0]
                lo2MP2_core = lo2core
            else:
                lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
                lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])

            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum() if MP2_bath_core.size else 0
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum() if MP2_bath_core.size else lo2core.shape[1]
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')

    else:
        log.info('========== NOT Read from MP2.TXT files ===========')
        if eta < 1:
            log.info('')
            log.info('========== eta < 1 ===========')
            log.info('constructing my ROMP2 bath')
            #======================= start new constructing ROMP2 bath ===========================#
            Ncore = ao2core.shape[1]
            Nvir =  ao2vir.shape[1]
        
            def semi_canonicalize(mf):
                fock = mf.get_fock()
                focka,fockb = fock.focka,fock.fockb
                mo = mf.mo_coeff
                coreidx = mf.mo_occ == 2
                viridx = mf.mo_occ == 0
                openidx = ~(coreidx|viridx)
                mo_focka = reduce(lib.dot, (mo.T, focka, mo))
                mo_fockb = reduce(lib.dot, (mo.T, fockb, mo))
                ea_occ,coeff_occa = np.linalg.eigh(mo_focka[coreidx|openidx,:][:,coreidx|openidx])
                ea_vir,coeff_vira = np.linalg.eigh(mo_focka[viridx,:][:,viridx])
                eb_occ,coeff_occb = np.linalg.eigh(mo_fockb[coreidx,:][:,coreidx])
                eb_vir,coeff_virb = np.linalg.eigh(mo_fockb[openidx|viridx,:][:,openidx|viridx])
                mo_coeff_occa = mo[:,coreidx|openidx]@coeff_occa
                mo_coeff_vira = mo[:,viridx]@coeff_vira
                mo_coeff_occb = mo[:,coreidx]@coeff_occb
                mo_coeff_virb = mo[:,openidx|viridx]@coeff_virb
                mo_a = np.hstack((mo_coeff_occa,mo_coeff_vira))
                mo_b = np.hstack((mo_coeff_occb,mo_coeff_virb))
                ea = np.concatenate((ea_occ,ea_vir))
                eb = np.concatenate((eb_occ,eb_vir))
                return (mo_a,mo_b), (ea,eb), (focka,fockb)
            
            semi_mo_coeff, semi_mo_energy, fockab = semi_canonicalize(mf)
            es_semi_mo_coeff, es_semi_mo_energy, es_fockab = semi_canonicalize(es_mf)
            
            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ
            
            occ_coeff = [semi_mo_coeff[i][:,mo_occ[i]>0] for i in range(2)]
            vir_coeff = [semi_mo_coeff[i][:,mo_occ[i]==0] for i in range(2)]
            es_occ_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]>0]) for i in range(2)]
            es_vir_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]==0]) for i in range(2)]
            
            occ_energy = [semi_mo_energy[i][mo_occ[i]>0] for i in range(2)]
            vir_energy = [semi_mo_energy[i][mo_occ[i]==0] for i in range(2)]
            es_occ_energy = [es_semi_mo_energy[i][es_mo_occ[i]>0] for i in range(2)]
            es_vir_energy = [es_semi_mo_energy[i][es_mo_occ[i]==0] for i in range(2)]
            
            def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
                log = logger.new_logger(mf, verbose)
            
                with_df = getattr(mf, 'with_df', None)
                assert( with_df is not None )
            
                if with_df._cderi is None:
                    log.debug('Caching ovL-type integrals directly')
                    if with_df.auxmol is None:
                        with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
                else:
                    log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
            
                assert (occ_coeff is not None and vir_coeff is not None)
            
                # determine incore or outcore
                nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
                nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
                naux = with_df.get_naoaux()
            
                if ovL is not None:
                    if isinstance(ovL, np.ndarray):
                        outcore = False
                    elif isinstance(ovL, str):
                        outcore = True
                    else:
                        log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                                  type(ovL))
                        raise TypeError
                else:
                    mem_now = mf.max_memory - lib.current_memory()[0]
                    mem_df = sum(nocc*nvir)*8/1024**2.
                    log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
                    
                    outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
                log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')
            
                if outcore:
                    eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                          ovL=ovL, ovL_to_save=ovL_to_save,
                                          verbose=log.verbose, stdout=log.stdout)
                else:
                    eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                         ovL=ovL,
                                         verbose=log.verbose, stdout=log.stdout)
                eris.build()
            
                return eris
            
            def get_t1(mf, fockab, occ_coeff=None, vir_coeff=None, occ_energy=None, vir_energy=None):
                focka, fockb = fockab
                gia = reduce(lib.dot, [occ_coeff[0].T, focka, vir_coeff[0]])
                gib = reduce(lib.dot, [occ_coeff[1].T, fockb, vir_coeff[1]])
                t1a = gia/lib.direct_sum('i-a->ia',occ_energy[0],vir_energy[0])
                t1b = gib/lib.direct_sum('i-a->ia',occ_energy[1],vir_energy[1])
                t1 = (t1a, t1b)
                return t1
            
            def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
            
                log = logger.new_logger(mf, verbose)
            
                assert (ao2mo is not None)
            
                nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
                nvirmax = max(nvir)
                assert (occ_energy is not None and vir_energy is not None)
            
                mem_avail = mf.max_memory - lib.current_memory()[0]
            
                if with_t2:
                    t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                          np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                          np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
                    t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
                    mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
                else:
                    t2 = None
                    t2_ptr = [lib.c_null_ptr()] * 3
            
                if mem_avail < 0:
                    log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
                    raise MemoryError
            
                libmp = lib.load_library('libmp')
                drv = libmp.MP2_contract_d
            
                # determine occ blksize
                if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
                    occ_blksize = nocc
                else:   # outcore ovL
                    # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
                    occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
                    occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]
            
                log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                          occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])
            
                cput1 = (logger.process_clock(), logger.perf_counter())
            
                for s in [0,1]:
                    s_t2 = 0 if s == 0 else 2
                    moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
                    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                        nocci = i1-i0
                        iaL = eris.get_occ_blk(s,i0,i1)
                        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                            noccj = j1-j0
                            if ibatch == jbatch:
                                jbL = iaL
                            else:
                                jbL = eris.get_occ_blk(s,j0,j1)
            
                            ed = np.zeros(1, dtype=np.float64)
                            ex = np.zeros(1, dtype=np.float64)
                            moeoo_block = np.asarray(
                                occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                            s2symm = 1
                            t2_ex = True
                            drv(
                                ed.ctypes.data_as(ctypes.c_void_p),
                                ex.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(s2symm),
                                iaL.ctypes.data_as(ctypes.c_void_p),
                                jbL.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(i0), ctypes.c_int(j0),
                                ctypes.c_int(nocci), ctypes.c_int(noccj),
                                ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                                moeoo_block.ctypes.data_as(ctypes.c_void_p),
                                moevv.ctypes.data_as(ctypes.c_void_p),
                                t2_ptr[s_t2], ctypes.c_int(t2_ex)
                            )
            
                            jbL = None
                        iaL = None
            
                        cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                                 *cput1)
                        
                # opposite spin
                sa, sb = 0, 1
                drv = libmp.MP2_OS_contract_d
                moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
                    nocci = i1-i0
                    iaL = eris.get_occ_blk(sa,i0,i1)
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                        noccj = j1-j0
                        jbL = eris.get_occ_blk(sb,j0,j1)
            
                        ed = np.zeros(1, dtype=np.float64)
                        moeoo_block = np.asarray(
                            occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                        drv(
                            ed.ctypes.data_as(ctypes.c_void_p),
                            iaL.ctypes.data_as(ctypes.c_void_p),
                            jbL.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(i0), ctypes.c_int(j0),
                            ctypes.c_int(nocci), ctypes.c_int(noccj),
                            ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                            ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                            ctypes.c_int(naux),
                            moeoo_block.ctypes.data_as(ctypes.c_void_p),
                            moevv.ctypes.data_as(ctypes.c_void_p),
                            t2_ptr[1]
                        )
            
                        jbL = None
                    iaL = None
            
                    cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                             *cput1)
            
                return t2
            
            def _gamma1_intermediates(mf, t1=None, t2=None, eris=None):
                assert (t1 is not None and t2 is not None)
                t1a, t1b = t1
                t2aa, t2ab, t2bb = t2
                nocca, noccb, nvira, nvirb = t2[1].shape
                
                dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
                dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
                dooa -= lib.einsum('ie,je->ij',t1a,t1a)
        
                doob  = lib.einsum('imef,jmef->ij', t2bb, t2bb) *-.5
                doob -= lib.einsum('mief,mjef->ij', t2ab, t2ab)
                doob -= lib.einsum('ie,je->ij',t1b,t1b)
        
            
                dvva  = lib.einsum('mnae,mnbe->ba', t2aa, t2aa) * .5
                dvva += lib.einsum('mnae,mnbe->ba', t2ab, t2ab)
                dvva += lib.einsum('ma,mb->ab',t1a,t1a)
        
                dvvb  = lib.einsum('mnae,mnbe->ba', t2bb, t2bb) * .5
                dvvb += lib.einsum('mnea,mneb->ba', t2ab, t2ab)
                dvvb += lib.einsum('ma,mb->ab',t1b,t1b)
                
                
                dooa += dooa.T
                doob += doob.T
                dvva += dvva.T
                dvvb += dvvb.T
                dooa *= 0.5
                doob *= 0.5
                dvva *= 0.5
                dvvb *= 0.5
                dooa[np.diag_indices(nocca)] += 1
                doob[np.diag_indices(noccb)] += 1
                
                dm1occ = [dooa,doob]
                dm1vir = [dvva,dvvb]
                return dm1occ, dm1vir
            
            eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
            eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)
            
            t_Ia = get_t1(mf, fockab, occ_coeff, es_vir_coeff, occ_energy, es_vir_energy)
            t_iA = get_t1(mf, fockab, es_occ_coeff, vir_coeff, es_occ_energy, vir_energy)
            t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose)
            t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose)
            
            D_IJ = _gamma1_intermediates(mf, t_Ia, t_IJab, eris_Ov)[0]
            D_AB = _gamma1_intermediates(mf, t_iA, t_ijAB, eris_oV)[1]
            
            S = mf.get_ovlp()
            D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
            D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])
            D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
            D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))
           
            #======================= END new constructing ROMP2 bath ===========================#

            np.savetxt(
                'D_MP2_core.txt',
                D_MP2_core,
                fmt='%.16e'
                        )

            np.savetxt(
                'D_MP2_vir.txt',
                D_MP2_vir,
                fmt='%.16e'
            )

            bins = np.array([10**-x for x in range(0,11)][::-1])
            eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
            log.info('This is eigvals_core %s', eigvals_core)
            histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
            log.info('Occupied BNO histogram')
            log.info('%s',histogram_core)
            log.info('')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
            log.info('This is eigvals_vir %s', eigvals_vir)
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            if ao:
                log.info('Bath expansion with AO-DMET')
                MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
            else:
                MP2_bath_core = (eigvals_core < 2 - eta)

            MP2_bath_vir = (eigvals_vir > eta)
            lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum()
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum()
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')

        else:
            G = eta
            log.info('')
            log.info('========== eta >= 1 ===========')
            log.info('constructing my ROMP2 bath')

            #======================= start new constructing ROMP2 bath ===========================#
            Ncore = ao2core.shape[1]
            Nvir =  ao2vir.shape[1]
        
            def semi_canonicalize(mf):
                fock = mf.get_fock()
                focka,fockb = fock.focka,fock.fockb
                mo = mf.mo_coeff
                coreidx = mf.mo_occ == 2
                viridx = mf.mo_occ == 0
                openidx = ~(coreidx|viridx)
                mo_focka = reduce(lib.dot, (mo.T, focka, mo))
                mo_fockb = reduce(lib.dot, (mo.T, fockb, mo))
                ea_occ,coeff_occa = np.linalg.eigh(mo_focka[coreidx|openidx,:][:,coreidx|openidx])
                ea_vir,coeff_vira = np.linalg.eigh(mo_focka[viridx,:][:,viridx])
                eb_occ,coeff_occb = np.linalg.eigh(mo_fockb[coreidx,:][:,coreidx])
                eb_vir,coeff_virb = np.linalg.eigh(mo_fockb[openidx|viridx,:][:,openidx|viridx])
                mo_coeff_occa = mo[:,coreidx|openidx]@coeff_occa
                mo_coeff_vira = mo[:,viridx]@coeff_vira
                mo_coeff_occb = mo[:,coreidx]@coeff_occb
                mo_coeff_virb = mo[:,openidx|viridx]@coeff_virb
                mo_a = np.hstack((mo_coeff_occa,mo_coeff_vira))
                mo_b = np.hstack((mo_coeff_occb,mo_coeff_virb))
                ea = np.concatenate((ea_occ,ea_vir))
                eb = np.concatenate((eb_occ,eb_vir))
                return (mo_a,mo_b), (ea,eb), (focka,fockb)

            semi_mo_coeff, semi_mo_energy, fockab = semi_canonicalize(mf)
            es_semi_mo_coeff, es_semi_mo_energy, es_fockab = semi_canonicalize(es_mf)

            mo_occ = mf.to_uhf().mo_occ
            es_mo_occ = es_mf.to_uhf().mo_occ

            occ_coeff = [semi_mo_coeff[i][:,mo_occ[i]>0] for i in range(2)]
            vir_coeff = [semi_mo_coeff[i][:,mo_occ[i]==0] for i in range(2)]
            es_occ_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]>0]) for i in range(2)]
            es_vir_coeff = [lib.dot(ao2eo, es_semi_mo_coeff[i][:,es_mo_occ[i]==0]) for i in range(2)]

            occ_energy = [semi_mo_energy[i][mo_occ[i]>0] for i in range(2)]
            vir_energy = [semi_mo_energy[i][mo_occ[i]==0] for i in range(2)]
            es_occ_energy = [es_semi_mo_energy[i][es_mo_occ[i]>0] for i in range(2)]
            es_vir_energy = [es_semi_mo_energy[i][es_mo_occ[i]==0] for i in range(2)]

            def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
                log = logger.new_logger(mf, verbose)

                with_df = getattr(mf, 'with_df', None)
                assert( with_df is not None )

                if with_df._cderi is None:
                    log.debug('Caching ovL-type integrals directly')
                    if with_df.auxmol is None:
                        with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
                else:
                    log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')

                assert (occ_coeff is not None and vir_coeff is not None)

                # determine incore or outcore
                nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
                nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
                naux = with_df.get_naoaux()

                if ovL is not None:
                    if isinstance(ovL, np.ndarray):
                        outcore = False
                    elif isinstance(ovL, str):
                        outcore = True
                    else:
                        log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                                  type(ovL))
                        raise TypeError
                else:
                    mem_now = mf.max_memory - lib.current_memory()[0]
                    mem_df = sum(nocc*nvir)*8/1024**2.
                    log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)

                    outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
                log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')

                if outcore:
                    eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                          ovL=ovL, ovL_to_save=ovL_to_save,
                                          verbose=log.verbose, stdout=log.stdout)
                else:
                    eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                         ovL=ovL,
                                         verbose=log.verbose, stdout=log.stdout)
                eris.build()

                return eris

            def get_t1(mf, fockab, occ_coeff=None, vir_coeff=None, occ_energy=None, vir_energy=None):
                focka, fockb = fockab
                gia = reduce(lib.dot, [occ_coeff[0].T, focka, vir_coeff[0]])
                gib = reduce(lib.dot, [occ_coeff[1].T, fockb, vir_coeff[1]])
                t1a = gia/lib.direct_sum('i-a->ia',occ_energy[0],vir_energy[0])
                t1b = gib/lib.direct_sum('i-a->ia',occ_energy[1],vir_energy[1])
                t1 = (t1a, t1b)
                return t1

            def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
            
                log = logger.new_logger(mf, verbose)

                assert (ao2mo is not None)

                nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
                nvirmax = max(nvir)
                assert (occ_energy is not None and vir_energy is not None)

                mem_avail = mf.max_memory - lib.current_memory()[0]

                if with_t2:
                    t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                          np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                          np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
                    t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
                    mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
                else:
                    t2 = None
                    t2_ptr = [lib.c_null_ptr()] * 3

                if mem_avail < 0:
                    log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
                    raise MemoryError

                libmp = lib.load_library('libmp')
                drv = libmp.MP2_contract_d

                # determine occ blksize
                if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
                    occ_blksize = nocc
                else:   # outcore ovL
                    # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
                    occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
                    occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]

                log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                          occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])

                cput1 = (logger.process_clock(), logger.perf_counter())

                for s in [0,1]:
                    s_t2 = 0 if s == 0 else 2
                    moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
                    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                        nocci = i1-i0
                        iaL = eris.get_occ_blk(s,i0,i1)
                        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                            noccj = j1-j0
                            if ibatch == jbatch:
                                jbL = iaL
                            else:
                                jbL = eris.get_occ_blk(s,j0,j1)

                            ed = np.zeros(1, dtype=np.float64)
                            ex = np.zeros(1, dtype=np.float64)
                            moeoo_block = np.asarray(
                                occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                            s2symm = 1
                            t2_ex = True
                            drv(
                                ed.ctypes.data_as(ctypes.c_void_p),
                                ex.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(s2symm),
                                iaL.ctypes.data_as(ctypes.c_void_p),
                                jbL.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(i0), ctypes.c_int(j0),
                                ctypes.c_int(nocci), ctypes.c_int(noccj),
                                ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                                moeoo_block.ctypes.data_as(ctypes.c_void_p),
                                moevv.ctypes.data_as(ctypes.c_void_p),
                                t2_ptr[s_t2], ctypes.c_int(t2_ex)
                            )

                            jbL = None
                        iaL = None

                        cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                                 *cput1)

                # opposite spin
                sa, sb = 0, 1
                drv = libmp.MP2_OS_contract_d
                moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
                for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
                    nocci = i1-i0
                    iaL = eris.get_occ_blk(sa,i0,i1)
                    for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                        noccj = j1-j0
                        jbL = eris.get_occ_blk(sb,j0,j1)

                        ed = np.zeros(1, dtype=np.float64)
                        moeoo_block = np.asarray(
                            occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                        drv(
                            ed.ctypes.data_as(ctypes.c_void_p),
                            iaL.ctypes.data_as(ctypes.c_void_p),
                            jbL.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(i0), ctypes.c_int(j0),
                            ctypes.c_int(nocci), ctypes.c_int(noccj),
                            ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                            ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                            ctypes.c_int(naux),
                            moeoo_block.ctypes.data_as(ctypes.c_void_p),
                            moevv.ctypes.data_as(ctypes.c_void_p),
                            t2_ptr[1]
                        )

                        jbL = None
                    iaL = None

                    cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                             *cput1)

                return t2

            def _gamma1_intermediates(mf, t1=None, t2=None, eris=None):
                assert (t1 is not None and t2 is not None)
                t1a, t1b = t1
                t2aa, t2ab, t2bb = t2
                nocca, noccb, nvira, nvirb = t2[1].shape

                dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
                dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
                dooa -= lib.einsum('ie,je->ij',t1a,t1a)

                doob  = lib.einsum('imef,jmef->ij', t2bb, t2bb) *-.5
                doob -= lib.einsum('mief,mjef->ij', t2ab, t2ab)
                doob -= lib.einsum('ie,je->ij',t1b,t1b)


                dvva  = lib.einsum('mnae,mnbe->ba', t2aa, t2aa) * .5
                dvva += lib.einsum('mnae,mnbe->ba', t2ab, t2ab)
                dvva += lib.einsum('ma,mb->ab',t1a,t1a)

                dvvb  = lib.einsum('mnae,mnbe->ba', t2bb, t2bb) * .5
                dvvb += lib.einsum('mnea,mneb->ba', t2ab, t2ab)
                dvvb += lib.einsum('ma,mb->ab',t1b,t1b)


                dooa += dooa.T
                doob += doob.T
                dvva += dvva.T
                dvvb += dvvb.T
                dooa *= 0.5
                doob *= 0.5
                dvva *= 0.5
                dvvb *= 0.5
                dooa[np.diag_indices(nocca)] += 1
                doob[np.diag_indices(noccb)] += 1

                dm1occ = [dooa,doob]
                dm1vir = [dvva,dvvb]
                return dm1occ, dm1vir

            eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
            eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)

            t_Ia = get_t1(mf, fockab, occ_coeff, es_vir_coeff, occ_energy, es_vir_energy)
            t_iA = get_t1(mf, fockab, es_occ_coeff, vir_coeff, es_occ_energy, vir_energy)
            t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose)
            t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose)

            D_IJ = _gamma1_intermediates(mf, t_Ia, t_IJab, eris_Ov)[0]
            D_AB = _gamma1_intermediates(mf, t_iA, t_ijAB, eris_oV)[1]

            S = mf.get_ovlp()
            D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
            D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])
            D_MP2_core = reduce(lib.dot,(ao2core.T, S, D_IJ_ao, S.T, ao2core))
            D_MP2_vir = reduce(lib.dot,(ao2vir.T, S, D_AB_ao, S.T, ao2vir))
           
            #======================= END new constructing ROMP2 bath ===========================#

            np.savetxt(
                'D_MP2_core.txt',
                D_MP2_core,
                fmt='%.16e'
                        )

            np.savetxt(
                'D_MP2_vir.txt',
                D_MP2_vir,
                fmt='%.16e'
            )

            bins = np.array([10**-x for x in range(0,11)][::-1])
            eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core)
            histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
            log.info('Occupied BNO histogram')
            log.info('%s',histogram_core)
            log.info('')

            eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir)
            histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
            log.info('Virtual BNO histogram')
            log.info('%s',histogram_vir)
            log.info('')

            eta = choose_eta_for_nbath(eigvals_core, eigvals_vir, G, tol=1e-12, strict=True)
            log.info("================This is new eta================")
            log.info(str(eta))

            if ao:
                log.info('Bath expansion with AO-DMET')
                MP2_bath_core = (eigvals_core < 2 - eta)|(eigvals_core > 2 + eta)
            else:
                MP2_bath_core = (eigvals_core < 2 - eta)

            MP2_bath_vir = (eigvals_vir > eta)
            lo2MP2_bath_core = lib.dot(lo2core, eigvecs_core[:,MP2_bath_core])
            lo2MP2_bath_vir = lib.dot(lo2vir, eigvecs_vir[:,MP2_bath_vir])
            lo2MP2_bath = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
            lo2MP2_core = lib.dot(lo2core, eigvecs_core[:,~MP2_bath_core])
            lo2MP2_vir = lib.dot(lo2vir, eigvecs_vir[:,~MP2_bath_vir])

            nbath_new_core = MP2_bath_core.sum()
            nbath_new_vir = MP2_bath_vir.sum()
            nbath_new = nbath_new_core + nbath_new_vir
            ncore_new = (~MP2_bath_core).sum()
            log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',nbath_new,nbath_new_core,nbath_new_vir)
            # log.info('Number of current frozen occupied orbitals = %s', ncore_new)
            log.info('')
            
    return lo2MP2_bath, lo2MP2_core, lo2MP2_vir

def make_histogram(values, bins, labels=True, binwidth=6, height=10, fill=":", show_number=False, invertx=True, rstrip=True):
    '''
    Modified from https://github.com/BoothGroup/Vayesta/blob/master/vayesta/core/bath/helper.py
    Original author: Max Nusspickel & Charles J. C. Scott
    '''
    hist = np.histogram(values, bins)[0]
    if invertx:
        bins, hist = bins[::-1], hist[::-1]
    hmax = hist.max()
    
    binwidths = [len(str(hval))-2 for hval in hist]

    width = binwidth * len(hist) + sum(binwidths)
    plot = np.zeros((height + show_number, width), dtype=str)
    plot[:] = " "
    if hmax > 0:
        for i, hval in enumerate(hist):
            colstart = i * binwidth + sum(binwidths[:i])
            colend = (i + 1) * binwidth + sum(binwidths[:(i+1)])
            barheight = int(np.rint(height * hval / hmax))
            if barheight == 0:
                continue
            # Top
            plot[-barheight, colstart + 1 : colend - 1] = "_"
            if show_number:
                number = " {:^{w}s}".format("%d" % hval, w=binwidth - 1 + binwidths[i])
                for idx, i in enumerate(range(colstart, colend)):
                    plot[-barheight - 1, i] = number[idx]

            if barheight == 1:
                continue
            # Fill
            if fill:
                plot[-barheight + 1 :, colstart + 1 : colend] = fill
            # Left/right border
            plot[-barheight + 1 :, colstart] = "|"
            plot[-barheight + 1 :, colend - 1] = "|"

    lines = ["".join(plot[r, :].tolist()) for r in range(height)]
    # Baseline
    lines.append("+" + ((width - 2) * "-") + "+")
    
    labelwides = np.hstack([6+np.array(binwidths)[1:],np.array([6])])
    if labels:
        lines += ["{:<{w}}".format("E-0", w=4) + "".join(["{:<{w}}".format("E-%d" % d, w=labelwides[i]) for i,d in enumerate(range(1, 11))])]

    if rstrip:
        lines = [line.rstrip() for line in lines]
    txt = "\n".join(lines)
    return txt

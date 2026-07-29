"""Ports Powell's NEWUOA trust-region optimizer from VCGLib's bundled header."""

from std.math import abs, atan, cos, max, min, sin, sqrt
from std.memory import Pointer, UnsafePointer
from std.sys.info import simd_width_of


comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime Objective = def(Int, Int, Int) thin abi("C") -> Float64
comptime W = simd_width_of[DType.float64]()
comptime TWO_PI = 6.283185307179586476925286766559
comptime DELTA_DECREASE = 0.5
comptime RHO_DECREASE = 0.5
# Keeps every workspace-size product within signed 64-bit Int arithmetic.
comptime MAX_SAFE_DIMENSION = 30_000


def _ptr(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def _objective(callback_addr: Int, n: Int, x: Ptr, user_data: Int) -> Float64:
    var opaque = UnsafePointer[NoneType, AnyOrigin[mut=True]](
        unsafe_from_address=callback_addr
    )
    var callback = Pointer(to=opaque).unsafe_bitcast[Objective]()[]
    return callback(n, Int(x), user_data)


# vcglib: wrap/newuoa/include/newuoa.h trsapp_
def _trsapp(
    n: Int,
    npt: Int,
    xopt: Ptr,
    xpt: Ptr,
    gq: Ptr,
    hq: Ptr,
    pq: Ptr,
    delta: Float64,
    step: Ptr,
    d: Ptr,
    g: Ptr,
    hd: Ptr,
    hs: Ptr,
) -> Float64:
    var tempa = 0.0
    var tempb = 0.0
    var shs = 0.0
    var sg = 0.0
    var bstep = 0.0
    var ggbeg = 0.0
    var gg = 0.0
    var qred = 0.0
    var dd = 0.0
    var crvmin = 0.0
    var ds = 0.0
    var ss = 0.0
    var dhd = 0.0
    var alpha = 0.0
    var qadd = 0.0
    var ggsav = 0.0
    var sgk = 0.0
    var angtest = 0.0
    var dg = 0.0
    var dhs = 0.0
    var cf = 0.0
    var qbeg = 0.0
    var qsav = 0.0
    var qmin = 0.0
    var qnew = 0.0
    var angle = 0.0
    var cth = 0.0
    var sth = 0.0
    var reduc = 0.0
    var ratio = 0.0
    var temp = 0.0
    var sumv = 0.0
    var delsq = delta * delta
    var iterc = 0
    var itermax = n
    var itersw = itermax
    var isave = 0
    var state = 170

    for i in range(n):
        d[i] = xopt[i]

    while True:
        if state == 170:
            for i in range(n):
                hd[i] = 0.0
            for k in range(npt):
                temp = 0.0
                for j in range(n):
                    temp += xpt[k + j * npt] * d[j]
                temp *= pq[k]
                for i in range(n):
                    hd[i] += temp * xpt[k + i * npt]
            var ih = 0
            for j in range(n):
                for i in range(j + 1):
                    if i < j:
                        hd[j] += hq[ih] * d[i]
                    hd[i] += hq[ih] * d[j]
                    ih += 1
            if iterc == 0:
                state = 20
            elif iterc <= itersw:
                state = 50
            else:
                state = 120
            continue

        if state == 20:
            qred = 0.0
            dd = 0.0
            for i in range(n):
                step[i] = 0.0
                hs[i] = 0.0
                g[i] = gq[i] + hd[i]
                d[i] = -g[i]
                dd += d[i] * d[i]
            crvmin = 0.0
            if dd == 0.0:
                return crvmin
            ds = 0.0
            ss = 0.0
            gg = dd
            ggbeg = gg
            state = 40
            continue

        if state == 40:
            iterc += 1
            temp = delsq - ss
            bstep = temp / (ds + sqrt(ds * ds + dd * temp))
            state = 170
            continue

        if state == 50:
            dhd = 0.0
            for j in range(n):
                dhd += d[j] * hd[j]
            alpha = bstep
            if dhd > 0.0:
                temp = dhd / dd
                if iterc == 1:
                    crvmin = temp
                crvmin = min(crvmin, temp)
                alpha = min(alpha, gg / dhd)
            qadd = alpha * (gg - 0.5 * alpha * dhd)
            qred += qadd
            ggsav = gg
            gg = 0.0
            for i in range(n):
                step[i] += alpha * d[i]
                hs[i] += alpha * hd[i]
                temp = g[i] + hs[i]
                gg += temp * temp
            if alpha < bstep:
                if (
                    qadd <= qred * 0.01
                    or gg <= ggbeg * 1.0e-4
                    or iterc == itermax
                ):
                    return crvmin
                temp = gg / ggsav
                dd = 0.0
                ds = 0.0
                ss = 0.0
                for i in range(n):
                    d[i] = temp * d[i] - g[i] - hs[i]
                    dd += d[i] * d[i]
                    ds += d[i] * step[i]
                    ss += step[i] * step[i]
                if ds <= 0.0:
                    return crvmin
                if ss < delsq:
                    state = 40
                    continue
            crvmin = 0.0
            itersw = iterc
            state = 90
            continue

        if state == 90:
            if gg <= ggbeg * 1.0e-4:
                return crvmin
            sg = 0.0
            shs = 0.0
            for i in range(n):
                sg += step[i] * g[i]
                shs += step[i] * hs[i]
            sgk = sg + shs
            angtest = sgk / sqrt(gg * delsq)
            if angtest <= -0.99:
                return crvmin
            iterc += 1
            temp = sqrt(delsq * gg - sgk * sgk)
            tempa = delsq / temp
            tempb = sgk / temp
            for i in range(n):
                d[i] = tempa * (g[i] + hs[i]) - tempb * step[i]
            state = 170
            continue

        dg = 0.0
        dhd = 0.0
        dhs = 0.0
        for i in range(n):
            dg += d[i] * g[i]
            dhd += hd[i] * d[i]
            dhs += hd[i] * step[i]
        cf = 0.5 * (shs - dhd)
        qbeg = sg + cf
        qsav = qbeg
        qmin = qbeg
        isave = 0
        temp = TWO_PI / 50.0
        qnew = qbeg
        for ii in range(1, 50):
            angle = Float64(ii) * temp
            cth = cos(angle)
            sth = sin(angle)
            qnew = (sg + cf * cth) * cth + (dg + dhs * cth) * sth
            if qnew < qmin:
                qmin = qnew
                isave = ii
                tempa = qsav
            elif ii == isave + 1:
                tempb = qnew
            qsav = qnew
        if isave == 0:
            tempa = qnew
        if isave == 49:
            tempb = qbeg
        angle = 0.0
        if tempa != tempb:
            tempa -= qmin
            tempb -= qmin
            angle = 0.5 * (tempa - tempb) / (tempa + tempb)
        angle = temp * (Float64(isave) + angle)
        cth = cos(angle)
        sth = sin(angle)
        reduc = qbeg - (sg + cf * cth) * cth - (dg + dhs * cth) * sth
        gg = 0.0
        for i in range(n):
            step[i] = cth * step[i] + sth * d[i]
            hs[i] = cth * hs[i] + sth * hd[i]
            temp = g[i] + hs[i]
            gg += temp * temp
        qred += reduc
        ratio = reduc / qred
        if iterc < itermax and ratio > 0.01:
            state = 90
            continue
        return crvmin


# vcglib: wrap/newuoa/include/newuoa.h update_
def _update(
    n: Int,
    npt: Int,
    bmat: Ptr,
    zmat: Ptr,
    idz_in: Int,
    ndim: Int,
    vlag: Ptr,
    beta: Float64,
    knew: Int,
    w: Ptr,
) -> Int:
    var idz = idz_in
    var tempb = 0.0
    var tempa = 0.0
    var temp = 0.0
    var alpha = 0.0
    var tau = 0.0
    var tausq = 0.0
    var denom = 0.0
    var scala = 0.0
    var scalb = 0.0
    var nptm = npt - n - 1
    var jl = 0
    for j in range(1, nptm):
        if j + 1 == idz:
            jl = idz - 1
        elif zmat[knew + j * npt] != 0.0:
            temp = sqrt(
                zmat[knew + jl * npt] * zmat[knew + jl * npt]
                + zmat[knew + j * npt] * zmat[knew + j * npt]
            )
            tempa = zmat[knew + jl * npt] / temp
            tempb = zmat[knew + j * npt] / temp
            for i in range(npt):
                temp = tempa * zmat[i + jl * npt] + tempb * zmat[i + j * npt]
                zmat[i + j * npt] = (
                    tempa * zmat[i + j * npt] - tempb * zmat[i + jl * npt]
                )
                zmat[i + jl * npt] = temp
            zmat[knew + j * npt] = 0.0
    tempa = zmat[knew]
    if idz >= 2:
        tempa = -tempa
    if jl > 0:
        tempb = zmat[knew + jl * npt]
    for i in range(npt):
        w[i] = tempa * zmat[i]
        if jl > 0:
            w[i] += tempb * zmat[i + jl * npt]
    alpha = w[knew]
    tau = vlag[knew]
    tausq = tau * tau
    denom = alpha * beta + tausq
    vlag[knew] -= 1.0
    var iflag = 0
    if jl == 0:
        temp = sqrt(abs(denom))
        tempb = tempa / temp
        tempa = tau / temp
        for i in range(npt):
            zmat[i] = tempa * zmat[i] - tempb * vlag[i]
        # The upstream test is preserved literally; sqrt makes temp nonnegative.
        if idz == 1 and temp < 0.0:
            idz = 2
        if idz >= 2 and temp >= 0.0:
            iflag = 1
    else:
        var ja = 0
        if beta >= 0.0:
            ja = jl
        var jb = jl - ja
        temp = zmat[knew + jb * npt] / denom
        tempa = temp * beta
        tempb = temp * tau
        temp = zmat[knew + ja * npt]
        scala = 1.0 / sqrt(abs(beta) * temp * temp + tausq)
        scalb = scala * sqrt(abs(denom))
        for i in range(npt):
            zmat[i + ja * npt] = scala * (
                tau * zmat[i + ja * npt] - temp * vlag[i]
            )
            zmat[i + jb * npt] = scalb * (
                zmat[i + jb * npt] - tempa * w[i] - tempb * vlag[i]
            )
        if denom <= 0.0:
            if beta < 0.0:
                idz += 1
            if beta >= 0.0:
                iflag = 1
    if iflag == 1:
        idz -= 1
        for i in range(npt):
            temp = zmat[i]
            zmat[i] = zmat[i + (idz - 1) * npt]
            zmat[i + (idz - 1) * npt] = temp
    for j in range(n):
        var jp = npt + j
        w[jp] = bmat[knew + j * ndim]
        tempa = (alpha * vlag[jp] - tau * w[jp]) / denom
        tempb = (-beta * w[jp] - tau * vlag[jp]) / denom
        for i in range(jp + 1):
            bmat[i + j * ndim] += tempa * vlag[i] + tempb * w[i]
            if i >= npt:
                bmat[jp + (i - npt) * ndim] = bmat[i + j * ndim]
    return idz


# vcglib: wrap/newuoa/include/newuoa.h biglag_
def _biglag(
    n: Int,
    npt: Int,
    xopt: Ptr,
    xpt: Ptr,
    bmat: Ptr,
    zmat: Ptr,
    idz: Int,
    ndim: Int,
    knew: Int,
    delta: Float64,
    d: Ptr,
    hcol: Ptr,
    gc: Ptr,
    gd: Ptr,
    s: Ptr,
    w: Ptr,
) -> Float64:
    var tempa = 0.0
    var tempb = 0.0
    var temp = 0.0
    var sumv = 0.0
    var sp = 0.0
    var ss = 0.0
    var dhd = 0.0
    var tau = 0.0
    var cth = 0.0
    var sth = 0.0
    var angle = 0.0
    var scale = 0.0
    var denom = 0.0
    var dd = 0.0
    var gg = 0.0
    var cf1 = 0.0
    var cf2 = 0.0
    var cf3 = 0.0
    var cf4 = 0.0
    var cf5 = 0.0
    var taubeg = 0.0
    var tauold = 0.0
    var taumax = 0.0
    var stepv = 0.0
    var delsq = delta * delta
    var nptm = npt - n - 1
    for k in range(npt):
        hcol[k] = 0.0
    for j in range(nptm):
        temp = zmat[knew + j * npt]
        if j + 1 < idz:
            temp = -temp
        for k in range(npt):
            hcol[k] += temp * zmat[k + j * npt]
    var alpha = hcol[knew]
    for i in range(n):
        d[i] = xpt[knew + i * npt] - xopt[i]
        gc[i] = bmat[knew + i * ndim]
        gd[i] = 0.0
        dd += d[i] * d[i]
    for k in range(npt):
        temp = 0.0
        sumv = 0.0
        for j in range(n):
            temp += xpt[k + j * npt] * xopt[j]
            sumv += xpt[k + j * npt] * d[j]
        temp *= hcol[k]
        sumv *= hcol[k]
        for i in range(n):
            gc[i] += temp * xpt[k + i * npt]
            gd[i] += sumv * xpt[k + i * npt]
    for i in range(n):
        gg += gc[i] * gc[i]
        sp += d[i] * gc[i]
        dhd += d[i] * gd[i]
    scale = delta / sqrt(dd)
    if sp * dhd < 0.0:
        scale = -scale
    temp = 0.0
    if sp * sp > dd * 0.99 * gg:
        temp = 1.0
    tau = scale * (abs(sp) + 0.5 * scale * abs(dhd))
    if gg * delsq < tau * 0.01 * tau:
        temp = 1.0
    for i in range(n):
        d[i] = scale * d[i]
        gd[i] = scale * gd[i]
        s[i] = gc[i] + temp * gd[i]
    for _ in range(n):
        dd = 0.0
        sp = 0.0
        ss = 0.0
        for i in range(n):
            dd += d[i] * d[i]
            sp += d[i] * s[i]
            ss += s[i] * s[i]
        temp = dd * ss - sp * sp
        if temp <= dd * 1.0e-8 * ss:
            return alpha
        denom = sqrt(temp)
        for i in range(n):
            s[i] = (dd * s[i] - sp * d[i]) / denom
            w[i] = 0.0
        for k in range(npt):
            sumv = 0.0
            for j in range(n):
                sumv += xpt[k + j * npt] * s[j]
            sumv *= hcol[k]
            for i in range(n):
                w[i] += sumv * xpt[k + i * npt]
        cf1 = 0.0
        cf2 = 0.0
        cf3 = 0.0
        cf4 = 0.0
        cf5 = 0.0
        for i in range(n):
            cf1 += s[i] * w[i]
            cf2 += d[i] * gc[i]
            cf3 += s[i] * gc[i]
            cf4 += d[i] * gd[i]
            cf5 += s[i] * gd[i]
        cf1 *= 0.5
        cf4 = 0.5 * cf4 - cf1
        taubeg = cf1 + cf2 + cf4
        taumax = taubeg
        tauold = taubeg
        var isave = 0
        temp = TWO_PI / 50.0
        for ii in range(1, 50):
            angle = Float64(ii) * temp
            cth = cos(angle)
            sth = sin(angle)
            tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth
            if abs(tau) > abs(taumax):
                taumax = tau
                isave = ii
                tempa = tauold
            elif ii == isave + 1:
                tempb = tau
            tauold = tau
        if isave == 0:
            tempa = tau
        if isave == 49:
            tempb = taubeg
        stepv = 0.0
        if tempa != tempb:
            tempa -= taumax
            tempb -= taumax
            stepv = 0.5 * (tempa - tempb) / (tempa + tempb)
        angle = temp * (Float64(isave) + stepv)
        cth = cos(angle)
        sth = sin(angle)
        tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth
        for i in range(n):
            d[i] = cth * d[i] + sth * s[i]
            gd[i] = cth * gd[i] + sth * w[i]
            s[i] = gc[i] + gd[i]
        if abs(tau) <= abs(taubeg) * 1.1:
            return alpha
    return alpha


# vcglib: wrap/newuoa/include/newuoa.h bigden_
def _bigden(
    n: Int,
    npt: Int,
    xopt: Ptr,
    xpt: Ptr,
    bmat: Ptr,
    zmat: Ptr,
    idz: Int,
    ndim: Int,
    kopt: Int,
    knew: Int,
    d: Ptr,
    w: Ptr,
    vlag: Ptr,
    beta_in: Float64,
    s: Ptr,
    wvec: Ptr,
    prod: Ptr,
) -> Float64:
    var den = InlineArray[Float64, 9](fill=0.0)
    var denex = InlineArray[Float64, 9](fill=0.0)
    var par = InlineArray[Float64, 9](fill=0.0)
    var beta = beta_in
    var dd = 0.0
    var ds = 0.0
    var ss = 0.0
    var tau = 0.0
    var sumv = 0.0
    var diff = 0.0
    var temp = 0.0
    var stepv = 0.0
    var alpha = 0.0
    var angle = 0.0
    var tempa = 0.0
    var tempb = 0.0
    var tempc = 0.0
    var ssden = 0.0
    var dtest = 0.0
    var xoptd = 0.0
    var xopts = 0.0
    var denold = 0.0
    var denmax = 0.0
    var densav = 0.0
    var dstemp = 0.0
    var sumold = 0.0
    var sstemp = 0.0
    var xoptsq = 0.0
    var nptm = npt - n - 1
    for k in range(npt):
        w[n + k] = 0.0
    for j in range(nptm):
        temp = zmat[knew + j * npt]
        if j + 1 < idz:
            temp = -temp
        for k in range(npt):
            w[n + k] += temp * zmat[k + j * npt]
    alpha = w[n + knew]
    for i in range(n):
        dd += d[i] * d[i]
        s[i] = xpt[knew + i * npt] - xopt[i]
        ds += d[i] * s[i]
        ss += s[i] * s[i]
        xoptsq += xopt[i] * xopt[i]
    if ds * ds > dd * 0.99 * ss:
        var ksav = knew
        dtest = ds * ds / ss
        for k in range(npt):
            if k != kopt:
                dstemp = 0.0
                sstemp = 0.0
                for i in range(n):
                    diff = xpt[k + i * npt] - xopt[i]
                    dstemp += d[i] * diff
                    sstemp += diff * diff
                if dstemp * dstemp / sstemp < dtest:
                    ksav = k
                    dtest = dstemp * dstemp / sstemp
                    ds = dstemp
                    ss = sstemp
        for i in range(n):
            s[i] = xpt[ksav + i * npt] - xopt[i]
    ssden = dd * ss - ds * ds
    var iterc = 0
    while True:
        iterc += 1
        temp = 1.0 / sqrt(ssden)
        xoptd = 0.0
        xopts = 0.0
        for i in range(n):
            s[i] = temp * (dd * s[i] - ds * d[i])
            xoptd += xopt[i] * d[i]
            xopts += xopt[i] * s[i]
        tempa = 0.5 * xoptd * xoptd
        tempb = 0.5 * xopts * xopts
        den[0] = dd * (xoptsq + 0.5 * dd) + tempa + tempb
        den[1] = 2.0 * xoptd * dd
        den[2] = 2.0 * xopts * dd
        den[3] = tempa - tempb
        den[4] = xoptd * xopts
        for i in range(5, 9):
            den[i] = 0.0
        for k in range(npt):
            tempa = 0.0
            tempb = 0.0
            tempc = 0.0
            for i in range(n):
                tempa += xpt[k + i * npt] * d[i]
                tempb += xpt[k + i * npt] * s[i]
                tempc += xpt[k + i * npt] * xopt[i]
            wvec[k] = 0.25 * (tempa * tempa + tempb * tempb)
            wvec[k + ndim] = tempa * tempc
            wvec[k + 2 * ndim] = tempb * tempc
            wvec[k + 3 * ndim] = 0.25 * (tempa * tempa - tempb * tempb)
            wvec[k + 4 * ndim] = 0.5 * tempa * tempb
        for i in range(n):
            var ip = npt + i
            wvec[ip] = 0.0
            wvec[ip + ndim] = d[i]
            wvec[ip + 2 * ndim] = s[i]
            wvec[ip + 3 * ndim] = 0.0
            wvec[ip + 4 * ndim] = 0.0
        for jc in range(5):
            var nw = npt
            if jc == 1 or jc == 2:
                nw = ndim
            for k in range(npt):
                prod[k + jc * ndim] = 0.0
            for j in range(nptm):
                sumv = 0.0
                for k in range(npt):
                    sumv += zmat[k + j * npt] * wvec[k + jc * ndim]
                if j + 1 < idz:
                    sumv = -sumv
                for k in range(npt):
                    prod[k + jc * ndim] += sumv * zmat[k + j * npt]
            if nw == ndim:
                for k in range(npt):
                    sumv = 0.0
                    for j in range(n):
                        sumv += bmat[k + j * ndim] * wvec[npt + j + jc * ndim]
                    prod[k + jc * ndim] += sumv
            for j in range(n):
                sumv = 0.0
                for i in range(nw):
                    sumv += bmat[i + j * ndim] * wvec[i + jc * ndim]
                prod[npt + j + jc * ndim] = sumv
        for k in range(ndim):
            sumv = 0.0
            for i in range(5):
                par[i] = 0.5 * prod[k + i * ndim] * wvec[k + i * ndim]
                sumv += par[i]
            den[0] = den[0] - par[0] - sumv
            tempa = prod[k] * wvec[k + ndim] + prod[k + ndim] * wvec[k]
            tempb = (
                prod[k + ndim] * wvec[k + 3 * ndim]
                + prod[k + 3 * ndim] * wvec[k + ndim]
            )
            tempc = (
                prod[k + 2 * ndim] * wvec[k + 4 * ndim]
                + prod[k + 4 * ndim] * wvec[k + 2 * ndim]
            )
            den[1] = den[1] - tempa - 0.5 * (tempb + tempc)
            den[5] -= 0.5 * (tempb - tempc)
            tempa = prod[k] * wvec[k + 2 * ndim] + prod[k + 2 * ndim] * wvec[k]
            tempb = (
                prod[k + ndim] * wvec[k + 4 * ndim]
                + prod[k + 4 * ndim] * wvec[k + ndim]
            )
            tempc = (
                prod[k + 2 * ndim] * wvec[k + 3 * ndim]
                + prod[k + 3 * ndim] * wvec[k + 2 * ndim]
            )
            den[2] = den[2] - tempa - 0.5 * (tempb - tempc)
            den[6] -= 0.5 * (tempb + tempc)
            tempa = prod[k] * wvec[k + 3 * ndim] + prod[k + 3 * ndim] * wvec[k]
            den[3] = den[3] - tempa - par[1] + par[2]
            tempa = prod[k] * wvec[k + 4 * ndim] + prod[k + 4 * ndim] * wvec[k]
            tempb = (
                prod[k + ndim] * wvec[k + 2 * ndim]
                + prod[k + 2 * ndim] * wvec[k + ndim]
            )
            den[4] = den[4] - tempa - 0.5 * tempb
            den[7] = den[7] - par[3] + par[4]
            tempa = (
                prod[k + 3 * ndim] * wvec[k + 4 * ndim]
                + prod[k + 4 * ndim] * wvec[k + 3 * ndim]
            )
            den[8] -= 0.5 * tempa
        sumv = 0.0
        for i in range(5):
            par[i] = 0.5 * prod[knew + i * ndim] * prod[knew + i * ndim]
            sumv += par[i]
        denex[0] = alpha * den[0] + par[0] + sumv
        tempa = 2.0 * prod[knew] * prod[knew + ndim]
        tempb = prod[knew + ndim] * prod[knew + 3 * ndim]
        tempc = prod[knew + 2 * ndim] * prod[knew + 4 * ndim]
        denex[1] = alpha * den[1] + tempa + tempb + tempc
        denex[5] = alpha * den[5] + tempb - tempc
        tempa = 2.0 * prod[knew] * prod[knew + 2 * ndim]
        tempb = prod[knew + ndim] * prod[knew + 4 * ndim]
        tempc = prod[knew + 2 * ndim] * prod[knew + 3 * ndim]
        denex[2] = alpha * den[2] + tempa + tempb - tempc
        denex[6] = alpha * den[6] + tempb + tempc
        tempa = 2.0 * prod[knew] * prod[knew + 3 * ndim]
        denex[3] = alpha * den[3] + tempa + par[1] - par[2]
        tempa = 2.0 * prod[knew] * prod[knew + 4 * ndim]
        denex[4] = (
            alpha * den[4] + tempa + prod[knew + ndim] * prod[knew + 2 * ndim]
        )
        denex[7] = alpha * den[7] + par[3] - par[4]
        denex[8] = (
            alpha * den[8] + prod[knew + 3 * ndim] * prod[knew + 4 * ndim]
        )
        sumv = denex[0] + denex[1] + denex[3] + denex[5] + denex[7]
        denold = sumv
        denmax = sumv
        var isave = 0
        temp = TWO_PI / 50.0
        par[0] = 1.0
        for i in range(1, 50):
            angle = Float64(i) * temp
            par[1] = cos(angle)
            par[2] = sin(angle)
            for j in range(4, 9, 2):
                par[j - 1] = par[1] * par[j - 3] - par[2] * par[j - 2]
                par[j] = par[1] * par[j - 2] + par[2] * par[j - 3]
            sumold = sumv
            sumv = 0.0
            for j in range(9):
                sumv += denex[j] * par[j]
            if abs(sumv) > abs(denmax):
                denmax = sumv
                isave = i
                tempa = sumold
            elif i == isave + 1:
                tempb = sumv
        if isave == 0:
            tempa = sumv
        if isave == 49:
            tempb = denold
        stepv = 0.0
        if tempa != tempb:
            tempa -= denmax
            tempb -= denmax
            stepv = 0.5 * (tempa - tempb) / (tempa + tempb)
        angle = temp * (Float64(isave) + stepv)
        par[1] = cos(angle)
        par[2] = sin(angle)
        for j in range(4, 9, 2):
            par[j - 1] = par[1] * par[j - 3] - par[2] * par[j - 2]
            par[j] = par[1] * par[j - 2] + par[2] * par[j - 3]
        beta = 0.0
        denmax = 0.0
        for j in range(9):
            beta += den[j] * par[j]
            denmax += denex[j] * par[j]
        for k in range(ndim):
            vlag[k] = 0.0
            for j in range(5):
                vlag[k] += prod[k + j * ndim] * par[j]
        tau = vlag[knew]
        dd = 0.0
        tempa = 0.0
        tempb = 0.0
        for i in range(n):
            d[i] = par[1] * d[i] + par[2] * s[i]
            w[i] = xopt[i] + d[i]
            dd += d[i] * d[i]
            tempa += d[i] * w[i]
            tempb += w[i] * w[i]
        if iterc >= n:
            break
        if iterc > 1:
            densav = max(densav, denold)
        if abs(denmax) <= abs(densav) * 1.1:
            break
        densav = denmax
        for i in range(n):
            temp = tempa * xopt[i] + tempb * d[i] - vlag[npt + i]
            s[i] = tau * bmat[knew + i * ndim] + alpha * temp
        for k in range(npt):
            sumv = 0.0
            for j in range(n):
                sumv += xpt[k + j * npt] * w[j]
            temp = (tau * w[n + k] - alpha * vlag[k]) * sumv
            for i in range(n):
                s[i] += temp * xpt[k + i * npt]
        ss = 0.0
        ds = 0.0
        for i in range(n):
            ss += s[i] * s[i]
            ds += d[i] * s[i]
        ssden = dd * ss - ds * ds
        if ssden < dd * 1.0e-8 * ss:
            break
    for k in range(ndim):
        w[k] = 0.0
        for j in range(5):
            w[k] += wvec[k + j * ndim] * par[j]
    vlag[kopt] += 1.0
    return beta


# vcglib: wrap/newuoa/include/newuoa.h newuob_
def _newuob(
    n: Int,
    npt: Int,
    x: Ptr,
    rhobeg: Float64,
    rhoend: Float64,
    maxfun: Int,
    xbase: Ptr,
    xopt: Ptr,
    xnew: Ptr,
    xpt: Ptr,
    fval: Ptr,
    gq: Ptr,
    hq: Ptr,
    pq: Ptr,
    bmat: Ptr,
    zmat: Ptr,
    ndim: Int,
    d: Ptr,
    vlag: Ptr,
    w: Ptr,
    callback_addr: Int,
    user_data: Int,
    meta: Ptr,
) -> Float64:
    var diffc = 0.0
    var ratio = 0.0
    var dnorm = 0.0
    var diffa = 0.0
    var diffb = 0.0
    var xoptsq = 0.0
    var f = 0.0
    var rho = 0.0
    var fbeg = 0.0
    var fopt = 0.0
    var xjpt = 0.0
    var xipt = 0.0
    var alpha = 0.0
    var dstep = 0.0
    var dx = 0.0
    var dsq = 0.0
    var sumv = 0.0
    var diff = 0.0
    var beta = 0.0
    var gisq = 0.0
    var temp = 0.0
    var suma = 0.0
    var sumb = 0.0
    var bsum = 0.0
    var gqsq = 0.0
    var sumz = 0.0
    var hdiag = 0.0
    var delta = 0.0
    var recip = 0.0
    var reciq = 0.0
    var fsave = 0.0
    var vquad = 0.0
    var tempq = 0.0
    var rhosq = 0.0
    var detrat = 0.0
    var crvmin = 0.0
    var distsq = 0.0
    var np = n + 1
    var nh = n * np // 2
    var nptm = npt - np
    var nftest = max(maxfun, 1)
    var nf = 0
    var nfm = 0
    var nfmm = 0
    var idz = 1
    var ipt = 0
    var jpt = 0
    var knew = 0
    var kopt = 0
    var ksave = 0
    var nfsav = 0
    var itest = 0
    var state = 50

    for j in range(n):
        xbase[j] = x[j]
        for k in range(npt):
            xpt[k + j * npt] = 0.0
        for i in range(ndim):
            bmat[i + j * ndim] = 0.0
    for ih in range(nh):
        hq[ih] = 0.0
    for k in range(npt):
        pq[k] = 0.0
        for j in range(nptm):
            zmat[k + j * npt] = 0.0
    rhosq = rhobeg * rhobeg
    recip = 1.0 / rhosq
    reciq = sqrt(0.5) / rhosq

    while True:
        if state == 50:
            nfm = nf
            nfmm = nf - n
            nf += 1
            if nfm <= 2 * n:
                if nfm >= 1 and nfm <= n:
                    xpt[nf - 1 + (nfm - 1) * npt] = rhobeg
                elif nfm > n:
                    xpt[nf - 1 + (nfmm - 1) * npt] = -rhobeg
            else:
                var itemp = (nfmm - 1) // n
                jpt = nfm - itemp * n - n
                ipt = jpt + itemp
                if ipt > n:
                    itemp = jpt
                    jpt = ipt - n
                    ipt = itemp
                xipt = rhobeg
                if fval[ipt + np - 1] < fval[ipt]:
                    xipt = -xipt
                xjpt = rhobeg
                if fval[jpt + np - 1] < fval[jpt]:
                    xjpt = -xjpt
                xpt[nf - 1 + (ipt - 1) * npt] = xipt
                xpt[nf - 1 + (jpt - 1) * npt] = xjpt
            for j in range(n):
                x[j] = xpt[nf - 1 + j * npt] + xbase[j]
            state = 310
            continue

        if state == 70:
            fval[nf - 1] = f
            if nf == 1:
                fbeg = f
                fopt = f
                kopt = 1
            elif f < fopt:
                fopt = f
                kopt = nf
            if nfm <= 2 * n:
                if nfm >= 1 and nfm <= n:
                    gq[nfm - 1] = (f - fbeg) / rhobeg
                    if npt < nf + n:
                        bmat[(nfm - 1) * ndim] = -1.0 / rhobeg
                        bmat[nf - 1 + (nfm - 1) * ndim] = 1.0 / rhobeg
                        bmat[npt + nfm - 1 + (nfm - 1) * ndim] = -0.5 * rhosq
                elif nfm > n:
                    bmat[nf - n - 1 + (nfmm - 1) * ndim] = 0.5 / rhobeg
                    bmat[nf - 1 + (nfmm - 1) * ndim] = -0.5 / rhobeg
                    zmat[(nfmm - 1) * npt] = -reciq - reciq
                    zmat[nf - n - 1 + (nfmm - 1) * npt] = reciq
                    zmat[nf - 1 + (nfmm - 1) * npt] = reciq
                    var ih = nfmm * (nfmm + 1) // 2
                    temp = (fbeg - f) / rhobeg
                    hq[ih - 1] = (gq[nfmm - 1] - temp) / rhobeg
                    gq[nfmm - 1] = 0.5 * (gq[nfmm - 1] + temp)
            else:
                var ih = ipt * (ipt - 1) // 2 + jpt
                if xipt < 0.0:
                    ipt += n
                if xjpt < 0.0:
                    jpt += n
                zmat[(nfmm - 1) * npt] = recip
                zmat[nf - 1 + (nfmm - 1) * npt] = recip
                zmat[ipt + (nfmm - 1) * npt] = -recip
                zmat[jpt + (nfmm - 1) * npt] = -recip
                hq[ih - 1] = (fbeg - fval[ipt] - fval[jpt] + f) / (xipt * xjpt)
            if nf < npt:
                state = 50
                continue
            rho = rhobeg
            delta = rho
            idz = 1
            diffa = 0.0
            diffb = 0.0
            itest = 0
            xoptsq = 0.0
            for i in range(n):
                xopt[i] = xpt[kopt - 1 + i * npt]
                xoptsq += xopt[i] * xopt[i]
            state = 90
            continue

        if state == 90:
            nfsav = nf
            state = 100
            continue

        if state == 100:
            knew = 0
            crvmin = _trsapp(
                n,
                npt,
                xopt,
                xpt,
                gq,
                hq,
                pq,
                delta,
                d,
                w,
                w + n,
                w + 2 * n,
                w + 3 * n,
            )
            dsq = 0.0
            for i in range(n):
                dsq += d[i] * d[i]
            dnorm = min(delta, sqrt(dsq))
            if dnorm < 0.5 * rho:
                knew = -1
                delta = DELTA_DECREASE * delta
                ratio = -1.0
                if delta <= rho * 1.5:
                    delta = rho
                if nf <= nfsav + 2:
                    state = 460
                    continue
                temp = crvmin * 0.125 * rho * rho
                if temp <= max(max(diffa, diffb), diffc):
                    state = 460
                    continue
                state = 490
                continue
            state = 120
            continue

        if state == 120:
            if dsq <= xoptsq * 0.001:
                tempq = xoptsq * 0.25
                for k in range(npt):
                    sumv = 0.0
                    for i in range(n):
                        sumv += xpt[k + i * npt] * xopt[i]
                    temp = pq[k] * sumv
                    sumv -= 0.5 * xoptsq
                    w[npt + k] = sumv
                    for i in range(n):
                        gq[i] += temp * xpt[k + i * npt]
                        xpt[k + i * npt] -= 0.5 * xopt[i]
                        vlag[i] = bmat[k + i * ndim]
                        w[i] = sumv * xpt[k + i * npt] + tempq * xopt[i]
                        var ip = npt + i
                        for j in range(i + 1):
                            bmat[ip + j * ndim] += (
                                vlag[i] * w[j] + w[i] * vlag[j]
                            )
                for k in range(nptm):
                    sumz = 0.0
                    for i in range(npt):
                        sumz += zmat[i + k * npt]
                        w[i] = w[npt + i] * zmat[i + k * npt]
                    for j in range(n):
                        sumv = tempq * sumz * xopt[j]
                        for i in range(npt):
                            sumv += w[i] * xpt[i + j * npt]
                        vlag[j] = sumv
                        if k + 1 < idz:
                            sumv = -sumv
                        for i in range(npt):
                            bmat[i + j * ndim] += sumv * zmat[i + k * npt]
                    for i in range(n):
                        var ip = i + npt
                        temp = vlag[i]
                        if k + 1 < idz:
                            temp = -temp
                        for j in range(i + 1):
                            bmat[ip + j * ndim] += temp * vlag[j]
                var ih = 0
                for j in range(n):
                    w[j] = 0.0
                    for k in range(npt):
                        w[j] += pq[k] * xpt[k + j * npt]
                        xpt[k + j * npt] -= 0.5 * xopt[j]
                    for i in range(j + 1):
                        if i < j:
                            gq[j] += hq[ih] * xopt[i]
                        gq[i] += hq[ih] * xopt[j]
                        hq[ih] += w[i] * xopt[j] + xopt[i] * w[j]
                        bmat[npt + i + j * ndim] = bmat[npt + j + i * ndim]
                        ih += 1
                for j in range(n):
                    xbase[j] += xopt[j]
                    xopt[j] = 0.0
                xoptsq = 0.0
            if knew > 0:
                alpha = _biglag(
                    n,
                    npt,
                    xopt,
                    xpt,
                    bmat,
                    zmat,
                    idz,
                    ndim,
                    knew - 1,
                    dstep,
                    d,
                    vlag,
                    vlag + npt,
                    w,
                    w + np - 1,
                    w + np + n - 1,
                )
            for k in range(npt):
                suma = 0.0
                sumb = 0.0
                sumv = 0.0
                for j in range(n):
                    suma += xpt[k + j * npt] * d[j]
                    sumb += xpt[k + j * npt] * xopt[j]
                    sumv += bmat[k + j * ndim] * d[j]
                w[k] = suma * (0.5 * suma + sumb)
                vlag[k] = sumv
            beta = 0.0
            for k in range(nptm):
                sumv = 0.0
                for i in range(npt):
                    sumv += zmat[i + k * npt] * w[i]
                if k + 1 < idz:
                    beta += sumv * sumv
                    sumv = -sumv
                else:
                    beta -= sumv * sumv
                for i in range(npt):
                    vlag[i] += sumv * zmat[i + k * npt]
            bsum = 0.0
            dx = 0.0
            for j in range(n):
                sumv = 0.0
                for i in range(npt):
                    sumv += w[i] * bmat[i + j * ndim]
                bsum += sumv * d[j]
                var jp = npt + j
                for k in range(n):
                    sumv += bmat[jp + k * ndim] * d[k]
                vlag[jp] = sumv
                bsum += sumv * d[j]
                dx += d[j] * xopt[j]
            beta = dx * dx + dsq * (xoptsq + dx + dx + 0.5 * dsq) + beta - bsum
            vlag[kopt - 1] += 1.0
            if knew > 0:
                temp = 1.0 + alpha * beta / (vlag[knew - 1] * vlag[knew - 1])
                if abs(temp) <= 0.8:
                    beta = _bigden(
                        n,
                        npt,
                        xopt,
                        xpt,
                        bmat,
                        zmat,
                        idz,
                        ndim,
                        kopt - 1,
                        knew - 1,
                        d,
                        w,
                        vlag,
                        beta,
                        xnew,
                        w + ndim,
                        w + 6 * ndim,
                    )
            state = 290
            continue

        if state == 290:
            var i = 0
            while i + W <= n:
                var xnew_vec = (
                    xopt.load[width=W](i)
                    + d.load[width=W](i)
                )
                xnew.store(i, xnew_vec)
                x.store(
                    i, xbase.load[width=W](i) + xnew_vec
                )
                i += W
            while i < n:
                xnew[i] = xopt[i] + d[i]
                x[i] = xbase[i] + xnew[i]
                i += 1
            nf += 1
            state = 310
            continue

        if state == 310:
            if nf > nftest:
                nf -= 1
                state = 530
                continue
            f = _objective(callback_addr, n, x, user_data)
            if nf <= npt:
                state = 70
                continue
            if knew == -1:
                state = 530
                continue
            vquad = 0.0
            var ih = 0
            for j in range(n):
                vquad += d[j] * gq[j]
                for i in range(j + 1):
                    temp = d[i] * xnew[j] + d[j] * xopt[i]
                    if i == j:
                        temp *= 0.5
                    vquad += temp * hq[ih]
                    ih += 1
            for k in range(npt):
                vquad += pq[k] * w[k]
            diff = f - fopt - vquad
            diffc = diffb
            diffb = diffa
            diffa = abs(diff)
            if dnorm > rho:
                nfsav = nf
            fsave = fopt
            if f < fopt:
                fopt = f
                xoptsq = 0.0
                for i in range(n):
                    xopt[i] = xnew[i]
                    xoptsq += xopt[i] * xopt[i]
            ksave = knew
            if knew > 0:
                state = 410
                continue
            if vquad >= 0.0:
                state = 530
                continue
            ratio = (f - fsave) / vquad
            if ratio <= 0.1:
                delta = 0.5 * dnorm
            elif ratio <= 0.7:
                delta = max(0.5 * delta, dnorm)
            else:
                delta = max(0.5 * delta, dnorm + dnorm)
            if delta <= rho * 1.5:
                delta = rho
            rhosq = max(0.1 * delta, rho)
            rhosq *= rhosq
            var ktemp = 0
            detrat = 0.0
            if f >= fsave:
                ktemp = kopt
                detrat = 1.0
            for k in range(1, npt + 1):
                hdiag = 0.0
                for j in range(nptm):
                    temp = 1.0
                    if j + 1 < idz:
                        temp = -1.0
                    hdiag += temp * (
                        zmat[k - 1 + j * npt] * zmat[k - 1 + j * npt]
                    )
                temp = abs(beta * hdiag + vlag[k - 1] * vlag[k - 1])
                distsq = 0.0
                for j in range(n):
                    var dist = xpt[k - 1 + j * npt] - xopt[j]
                    distsq += dist * dist
                if distsq > rhosq:
                    var scale = distsq / rhosq
                    temp *= scale * scale * scale
                if temp > detrat and k != ktemp:
                    detrat = temp
                    knew = k
            if knew == 0:
                state = 460
                continue
            state = 410
            continue

        if state == 410:
            idz = _update(
                n,
                npt,
                bmat,
                zmat,
                idz,
                ndim,
                vlag,
                beta,
                knew - 1,
                w,
            )
            fval[knew - 1] = f
            var ih = 0
            for i in range(n):
                temp = pq[knew - 1] * xpt[knew - 1 + i * npt]
                for j in range(i + 1):
                    hq[ih] += temp * xpt[knew - 1 + j * npt]
                    ih += 1
            pq[knew - 1] = 0.0
            for j in range(nptm):
                temp = diff * zmat[knew - 1 + j * npt]
                if j + 1 < idz:
                    temp = -temp
                for k in range(npt):
                    pq[k] += temp * zmat[k + j * npt]
            gqsq = 0.0
            for i in range(n):
                gq[i] += diff * bmat[knew - 1 + i * ndim]
                gqsq += gq[i] * gq[i]
                xpt[knew - 1 + i * npt] = xnew[i]
            if ksave == 0 and delta == rho:
                if abs(ratio) > 0.01:
                    itest = 0
                else:
                    for k in range(npt):
                        vlag[k] = fval[k] - fval[kopt - 1]
                    gisq = 0.0
                    for i in range(n):
                        sumv = 0.0
                        for k in range(npt):
                            sumv += bmat[k + i * ndim] * vlag[k]
                        gisq += sumv * sumv
                        w[i] = sumv
                    itest += 1
                    if gqsq < gisq * 100.0:
                        itest = 0
                    if itest >= 3:
                        for i in range(n):
                            gq[i] = w[i]
                        for ih2 in range(nh):
                            hq[ih2] = 0.0
                        for j in range(nptm):
                            w[j] = 0.0
                            for k in range(npt):
                                w[j] += vlag[k] * zmat[k + j * npt]
                            if j + 1 < idz:
                                w[j] = -w[j]
                        for k in range(npt):
                            pq[k] = 0.0
                            for j in range(nptm):
                                pq[k] += zmat[k + j * npt] * w[j]
                        itest = 0
            if f < fsave:
                kopt = knew
            if f <= fsave + 0.1 * vquad or ksave > 0:
                state = 100
                continue
            knew = 0
            state = 460
            continue

        if state == 460:
            distsq = delta * 4.0 * delta
            for k in range(1, npt + 1):
                sumv = 0.0
                for j in range(n):
                    temp = xpt[k - 1 + j * npt] - xopt[j]
                    sumv += temp * temp
                if sumv > distsq:
                    knew = k
                    distsq = sumv
            if knew > 0:
                dstep = max(
                    min(0.1 * sqrt(distsq), 0.5 * delta),
                    rho,
                )
                dsq = dstep * dstep
                state = 120
                continue
            if ratio > 0.0 or max(delta, dnorm) > rho:
                state = 100
                continue
            state = 490
            continue

        if state == 490:
            if rho > rhoend:
                delta = 0.5 * rho
                ratio = rho / rhoend
                if ratio <= 16.0:
                    rho = rhoend
                elif ratio <= 250.0:
                    rho = sqrt(ratio) * rhoend
                else:
                    rho = RHO_DECREASE * rho
                delta = max(delta, rho)
                state = 90
                continue
            if knew == -1:
                state = 290
                continue
            state = 530
            continue

        if fopt <= f:
            var i = 0
            while i + W <= n:
                x.store(
                    i,
                    xbase.load[width=W](i)
                    + xopt.load[width=W](i),
                )
                i += W
            while i < n:
                x[i] = xbase[i] + xopt[i]
                i += 1
            f = fopt
        meta[1] = Float64(nf)
        return f


# vcglib: wrap/newuoa/include/newuoa.h newuoa_ / min_newuoa
def _workspace_size(n: Int, npt: Int) -> Int:
    return (npt + 13) * (npt + n) + 3 * n * (n + 3) // 2 + 11


@export("mnu_workspace_size")
def mnu_workspace_size(n: Int, npt: Int) abi("C") -> Int:
    if n < 2 or n > MAX_SAFE_DIMENSION:
        return 0
    if npt < n + 2 or npt > (n + 1) * (n + 2) // 2:
        return 0
    return _workspace_size(n, npt)


@export("mnu_minimize_f64")
def mnu_minimize_f64(
    n: Int,
    npt: Int,
    x_addr: Int,
    x_len: Int,
    rhobeg: Float64,
    rhoend: Float64,
    maxfun: Int,
    workspace_addr: Int,
    workspace_len: Int,
    callback_addr: Int,
    user_data: Int,
    meta_addr: Int,
    meta_len: Int,
) abi("C") -> Int:
    if n < 2:
        return -1
    if n > MAX_SAFE_DIMENSION:
        return -6
    if npt < n + 2 or npt > (n + 1) * (n + 2) // 2:
        return -2
    if rhobeg <= 0.0 or rhoend <= 0.0 or rhoend > rhobeg:
        return -3
    if (
        x_addr == 0
        or workspace_addr == 0
        or callback_addr == 0
        or meta_addr == 0
    ):
        return -4
    if x_len < n or workspace_len < _workspace_size(n, npt) or meta_len < 3:
        return -5
    var x = _ptr(x_addr)
    var work = _ptr(workspace_addr)
    var meta = _ptr(meta_addr)
    var np = n + 1
    var nptm = npt - np
    var ndim = npt + n
    var ixb = 0
    var ixo = ixb + n
    var ixn = ixo + n
    var ixp = ixn + n
    var ifv = ixp + n * npt
    var igq = ifv + npt
    var ihq = igq + n
    var ipq = ihq + n * np // 2
    var ibmat = ipq + npt
    var izmat = ibmat + ndim * n
    var id = izmat + npt * nptm
    var ivl = id + n
    var iw = ivl + ndim
    meta[0] = 0.0
    meta[1] = 0.0
    meta[2] = 0.0
    var value = _newuob(
        n,
        npt,
        x,
        rhobeg,
        rhoend,
        maxfun,
        work + ixb,
        work + ixo,
        work + ixn,
        work + ixp,
        work + ifv,
        work + igq,
        work + ihq,
        work + ipq,
        work + ibmat,
        work + izmat,
        ndim,
        work + id,
        work + ivl,
        work + iw,
        callback_addr,
        user_data,
        meta,
    )
    meta[0] = value
    return 0


subroutine func(nvar,ncon,x,f,g)

    implicit      none

    integer       nvar
    integer       ncon
    real*8        x(nvar)
    real*8        f
    real*8        g(ncon)

    integer       d
    integer       duuuu
    integer       deeee
    integer       dssss
    integer       drrrr
    integer       ny

    parameter     (d=124)
    parameter     (duuuu=91)
    parameter     (deeee=96)
    parameter     (dssss=84)
    parameter     (drrrr=44)
    parameter     (ny=68)

    integer       uuuu2opt(duuuu)
    integer       eeee2opt(deeee)
    integer       ssss2opt(dssss)
    integer       rrrr2opt(drrrr)
    integer       ssss(ny)

    integer       i
    integer       k
    real*8        xuuuu(duuuu)
    real*8        xeeee(deeee)
    real*8        xssss(dssss)
    real*8        xrrrr(drrrr)
    real*8        mmmm(d)
    real*8        aaaa(d)
    real*8        xl(nvar)
    real*8        xu(nvar)
    real*8        gu(ncon)
    real*8        xxxxx(nvar)

    logical       first

    data          first /.true./

    SAVE

    if (first) then

        first = .false.

        call get_uuuu2opt( uuuu2opt )
        call get_eeee2opt( eeee2opt )
        call get_ssss2opt( ssss2opt )
        call get_rrrr2opt( rrrr2opt )
        call get_mmmm( mmmm )
        call get_aaaa( aaaa )
        call get_ssss( ssss )

    endif

    call get_bbbb(xl, xu, gu)

    do i=1,nvar
        xxxxx(i) = xl(i) + x(i) * (xu(i) - xl(i))
    enddo

    do k=1,duuuu
        i = uuuu2opt(k)
        if (i > 0) then
            xuuuu(k) = xxxxx(i)
        else
            xuuuu(k) = aaaa(-i)
        endif
    enddo

    do k=1,deeee
        i = eeee2opt(k)
        if (i > 0) then
            xeeee(k) = xxxxx(i)
        else
            xeeee(k) = aaaa(-i)
        endif
    enddo

    do k=1,dssss
        i = ssss2opt(k)
        if (i > 0) then
            xssss(k) = xxxxx(i)
        else
            xssss(k) = aaaa(-i)
        endif
    enddo

    do k=1,drrrr
        xrrrr(k) = xxxxx( rrrr2opt(k) )
    enddo

    f = 0.0d0
    do i=1,d
        f = f + mmmm(i) * xxxxx(i)
    enddo

    call g1(xuuuu,g(1))
    call g2(xeeee,g(2))
    call g3(xeeee,g(3))
    call g4(xeeee,g(4))
    call g5(xeeee,g(5))
    call g6(xeeee,g(6))
    call g7(xssss,g(7))
    call g8(xssss,g(8))
    call g9(xssss,g(9))
    call g10(xxxxx,g(10))
    call g11(xxxxx,g(11))
    call g12(xxxxx,g(12))
    call g13(xxxxx,g(13))
    call g14(xxxxx,g(14))
    call g15(xxxxx,g(15))
    call g16(xxxxx,g(16))
    call g17(xxxxx,g(17))
    call g18(xxxxx,g(18))
    call g19(xxxxx,g(19))
    call g20(xxxxx,g(20))
    call g21(xxxxx,g(21))
    call g22(xxxxx,g(22))
    call g23(xxxxx,g(23))
    call g24(xxxxx,g(24))
    call g25(xxxxx,g(25))
    call g26(xxxxx,g(26))
    call g27(xxxxx,g(27))
    call g28(xxxxx,g(28))
    call g29(xxxxx,g(29))
    call g30(xxxxx,g(30))
    call g31(xxxxx,g(31))
    call g32(xxxxx,g(32))
    call g33(xxxxx,g(33))
    call g34(xxxxx,g(34))
    call g35(xxxxx,g(35))
    call g36(xxxxx,g(36))
    call g37(xxxxx,g(37))
    call g38(xxxxx,g(38))
    call g39(xxxxx,g(39))
    call g40(xxxxx,g(40))
    call g41(xxxxx,g(41))
    call g42(xxxxx,g(42))
    call g43(xxxxx,g(43))
    call g44(xxxxx,g(44))
    call g45(xxxxx,g(45))
    call g46(xxxxx,g(46))
    call g47(xxxxx,g(47))
    call g48(xxxxx,g(48))
    call g49(xxxxx,g(49))
    call g50(xxxxx,g(50))
    call g51(xxxxx,g(51))
    call g52(xxxxx,g(52))
    call g53(xxxxx,g(53))
    call g54(xxxxx,g(54))
    call g55(xxxxx,g(55))
    call g56(xrrrr,g(56))
    call g57(xrrrr,g(57))
    call g58(xrrrr,g(58))
    call g59(xrrrr,g(59))
    call g60(xrrrr,g(60))
    call g61(xrrrr,g(61))
    call g62(xrrrr,g(62))
    call g63(xrrrr,g(63))
    call g64(xrrrr,g(64))
    call g65(xrrrr,g(65))
    call g66(xrrrr,g(66))
    call g67(xrrrr,g(67))
    call g68(xrrrr,g(68))

    do i=1,ny
        g(i) = ssss(i) * g(i)
        g(i) = ( g(i) - gu(i) ) / abs(gu(i))
    enddo

    return

end
    
!**********************************************************************

subroutine get_uuuu2opt( uuuu2opt )

    implicit none
    
    integer  uuuu2opt(91)

    uuuu2opt(1) = 1
    uuuu2opt(2) = 2
    uuuu2opt(3) = 3
    uuuu2opt(4) = 4
    uuuu2opt(5) = 5
    uuuu2opt(6) = 6
    uuuu2opt(7) = 7
    uuuu2opt(8) = 8
    uuuu2opt(9) = 9
    uuuu2opt(10) = 10
    uuuu2opt(11) = 11
    uuuu2opt(12) = 28
    uuuu2opt(13) = 29
    uuuu2opt(14) = 30
    uuuu2opt(15) = 31
    uuuu2opt(16) = 32
    uuuu2opt(17) = 33
    uuuu2opt(18) = 34
    uuuu2opt(19) = 35
    uuuu2opt(20) = 36
    uuuu2opt(21) = 37
    uuuu2opt(22) = 38
    uuuu2opt(23) = 39
    uuuu2opt(24) = 40
    uuuu2opt(25) = 42
    uuuu2opt(26) = 44
    uuuu2opt(27) = 45
    uuuu2opt(28) = 49
    uuuu2opt(29) = 53
    uuuu2opt(30) = 54
    uuuu2opt(31) = 55
    uuuu2opt(32) = 56
    uuuu2opt(33) = 57
    uuuu2opt(34) = 58
    uuuu2opt(35) = 59
    uuuu2opt(36) = 60
    uuuu2opt(37) = 61
    uuuu2opt(38) = 62
    uuuu2opt(39) = 63
    uuuu2opt(40) = 64
    uuuu2opt(41) = 65
    uuuu2opt(42) = 66
    uuuu2opt(43) = 67
    uuuu2opt(44) = 72
    uuuu2opt(45) = 73
    uuuu2opt(46) = 85
    uuuu2opt(47) = 87
    uuuu2opt(48) = 94
    uuuu2opt(49) = 114
    uuuu2opt(50) = 115
    uuuu2opt(51) = 116
    uuuu2opt(52) = 117
    uuuu2opt(53) = 118
    uuuu2opt(54) = 119
    uuuu2opt(55) = 120
    uuuu2opt(56) = 121
    uuuu2opt(57) = 122
    uuuu2opt(58) = 123
    uuuu2opt(59) = 124
    uuuu2opt(60) = -1
    uuuu2opt(61) = -2
    uuuu2opt(62) = -4
    uuuu2opt(63) = -7
    uuuu2opt(64) = -8
    uuuu2opt(65) = -11
    uuuu2opt(66) = -29
    uuuu2opt(67) = -30
    uuuu2opt(68) = -31
    uuuu2opt(69) = -33
    uuuu2opt(70) = -37
    uuuu2opt(71) = -39
    uuuu2opt(72) = -42
    uuuu2opt(73) = -44
    uuuu2opt(74) = -45
    uuuu2opt(75) = -49
    uuuu2opt(76) = -54
    uuuu2opt(77) = -55
    uuuu2opt(78) = -56
    uuuu2opt(79) = -57
    uuuu2opt(80) = -58
    uuuu2opt(81) = -60
    uuuu2opt(82) = -61
    uuuu2opt(83) = -62
    uuuu2opt(84) = -63
    uuuu2opt(85) = -65
    uuuu2opt(86) = -67
    uuuu2opt(87) = -73
    uuuu2opt(88) = -85
    uuuu2opt(89) = -87
    uuuu2opt(90) = -114
    uuuu2opt(91) = -115

    return

end

!**********************************************************************

subroutine get_eeee2opt( eeee2opt )

    implicit none
    
    integer  eeee2opt(96)

    eeee2opt(1) = 1
    eeee2opt(2) = 2
    eeee2opt(3) = 3
    eeee2opt(4) = 4
    eeee2opt(5) = 5
    eeee2opt(6) = 6
    eeee2opt(7) = 7
    eeee2opt(8) = 8
    eeee2opt(9) = 9
    eeee2opt(10) = 10
    eeee2opt(11) = 11
    eeee2opt(12) = 28
    eeee2opt(13) = 29
    eeee2opt(14) = 30
    eeee2opt(15) = 31
    eeee2opt(16) = 32
    eeee2opt(17) = 33
    eeee2opt(18) = 34
    eeee2opt(19) = 35
    eeee2opt(20) = 36
    eeee2opt(21) = 37
    eeee2opt(22) = 38
    eeee2opt(23) = 39
    eeee2opt(24) = 40
    eeee2opt(25) = 42
    eeee2opt(26) = 44
    eeee2opt(27) = 45
    eeee2opt(28) = 49
    eeee2opt(29) = 53
    eeee2opt(30) = 54
    eeee2opt(31) = 55
    eeee2opt(32) = 56
    eeee2opt(33) = 57
    eeee2opt(34) = 58
    eeee2opt(35) = 59
    eeee2opt(36) = 60
    eeee2opt(37) = 61
    eeee2opt(38) = 62
    eeee2opt(39) = 63
    eeee2opt(40) = 64
    eeee2opt(41) = 65
    eeee2opt(42) = 66
    eeee2opt(43) = 67
    eeee2opt(44) = 72
    eeee2opt(45) = 73
    eeee2opt(46) = 85
    eeee2opt(47) = 87
    eeee2opt(48) = 94
    eeee2opt(49) = 114
    eeee2opt(50) = 115
    eeee2opt(51) = 116
    eeee2opt(52) = 117
    eeee2opt(53) = 118
    eeee2opt(54) = 119
    eeee2opt(55) = 120
    eeee2opt(56) = 121
    eeee2opt(57) = 122
    eeee2opt(58) = 123
    eeee2opt(59) = 124
    eeee2opt(60) = -1
    eeee2opt(61) = -2
    eeee2opt(62) = -3
    eeee2opt(63) = -4
    eeee2opt(64) = -5
    eeee2opt(65) = -6
    eeee2opt(66) = -7
    eeee2opt(67) = -8
    eeee2opt(68) = -9
    eeee2opt(69) = -11
    eeee2opt(70) = -29
    eeee2opt(71) = -30
    eeee2opt(72) = -32
    eeee2opt(73) = -34
    eeee2opt(74) = -36
    eeee2opt(75) = -37
    eeee2opt(76) = -38
    eeee2opt(77) = -42
    eeee2opt(78) = -44
    eeee2opt(79) = -49
    eeee2opt(80) = -53
    eeee2opt(81) = -54
    eeee2opt(82) = -55
    eeee2opt(83) = -56
    eeee2opt(84) = -58
    eeee2opt(85) = -59
    eeee2opt(86) = -60
    eeee2opt(87) = -61
    eeee2opt(88) = -62
    eeee2opt(89) = -63
    eeee2opt(90) = -64
    eeee2opt(91) = -65
    eeee2opt(92) = -67
    eeee2opt(93) = -73
    eeee2opt(94) = -87
    eeee2opt(95) = -114
    eeee2opt(96) = -115

    return

end

!**********************************************************************

subroutine get_ssss2opt( ssss2opt )

    implicit none

    integer  ssss2opt(84)

    ssss2opt(1) = 4
    ssss2opt(2) = 5
    ssss2opt(3) = 11
    ssss2opt(4) = 12
    ssss2opt(5) = 13
    ssss2opt(6) = 16
    ssss2opt(7) = 19
    ssss2opt(8) = 20
    ssss2opt(9) = 22
    ssss2opt(10) = 23
    ssss2opt(11) = 24
    ssss2opt(12) = 26
    ssss2opt(13) = 27
    ssss2opt(14) = 31
    ssss2opt(15) = 32
    ssss2opt(16) = 41
    ssss2opt(17) = 42
    ssss2opt(18) = 43
    ssss2opt(19) = 44
    ssss2opt(20) = 45
    ssss2opt(21) = 46
    ssss2opt(22) = 47
    ssss2opt(23) = 48
    ssss2opt(24) = 49
    ssss2opt(25) = 50
    ssss2opt(26) = 51
    ssss2opt(27) = 52
    ssss2opt(28) = 69
    ssss2opt(29) = 70
    ssss2opt(30) = 71
    ssss2opt(31) = 72
    ssss2opt(32) = 73
    ssss2opt(33) = 74
    ssss2opt(34) = 75
    ssss2opt(35) = 76
    ssss2opt(36) = 77
    ssss2opt(37) = 79
    ssss2opt(38) = 80
    ssss2opt(39) = 81
    ssss2opt(40) = 82
    ssss2opt(41) = 83
    ssss2opt(42) = 84
    ssss2opt(43) = 85
    ssss2opt(44) = 86
    ssss2opt(45) = 87
    ssss2opt(46) = 88
    ssss2opt(47) = 89
    ssss2opt(48) = 90
    ssss2opt(49) = 91
    ssss2opt(50) = 94
    ssss2opt(51) = 99
    ssss2opt(52) = 101
    ssss2opt(53) = 102
    ssss2opt(54) = 104
    ssss2opt(55) = 105
    ssss2opt(56) = 106
    ssss2opt(57) = 107
    ssss2opt(58) = 109
    ssss2opt(59) = 110
    ssss2opt(60) = -4
    ssss2opt(61) = -5
    ssss2opt(62) = -11
    ssss2opt(63) = -23
    ssss2opt(64) = -24
    ssss2opt(65) = -26
    ssss2opt(66) = -42
    ssss2opt(67) = -43
    ssss2opt(68) = -44
    ssss2opt(69) = -46
    ssss2opt(70) = -47
    ssss2opt(71) = -48
    ssss2opt(72) = -49
    ssss2opt(73) = -50
    ssss2opt(74) = -51
    ssss2opt(75) = -69
    ssss2opt(76) = -70
    ssss2opt(77) = -73
    ssss2opt(78) = -77
    ssss2opt(79) = -83
    ssss2opt(80) = -86
    ssss2opt(81) = -87
    ssss2opt(82) = -89
    ssss2opt(83) = -101
    ssss2opt(84) = -110

    return

end

!**********************************************************************

subroutine get_rrrr2opt( rrrr2opt )

    implicit none

    integer  rrrr2opt(44)

    rrrr2opt(1) = 11
    rrrr2opt(2) = 14
    rrrr2opt(3) = 15
    rrrr2opt(4) = 17
    rrrr2opt(5) = 18
    rrrr2opt(6) = 21
    rrrr2opt(7) = 22
    rrrr2opt(8) = 25
    rrrr2opt(9) = 26
    rrrr2opt(10) = 27
    rrrr2opt(11) = 41
    rrrr2opt(12) = 42
    rrrr2opt(13) = 50
    rrrr2opt(14) = 51
    rrrr2opt(15) = 52
    rrrr2opt(16) = 69
    rrrr2opt(17) = 71
    rrrr2opt(18) = 72
    rrrr2opt(19) = 76
    rrrr2opt(20) = 77
    rrrr2opt(21) = 78
    rrrr2opt(22) = 79
    rrrr2opt(23) = 80
    rrrr2opt(24) = 81
    rrrr2opt(25) = 90
    rrrr2opt(26) = 91
    rrrr2opt(27) = 92
    rrrr2opt(28) = 96
    rrrr2opt(29) = 98
    rrrr2opt(30) = 99
    rrrr2opt(31) = 100
    rrrr2opt(32) = 101
    rrrr2opt(33) = 102
    rrrr2opt(34) = 103
    rrrr2opt(35) = 104
    rrrr2opt(36) = 105
    rrrr2opt(37) = 106
    rrrr2opt(38) = 107
    rrrr2opt(39) = 108
    rrrr2opt(40) = 109
    rrrr2opt(41) = 110
    rrrr2opt(42) = 111
    rrrr2opt(43) = 112
    rrrr2opt(44) = 113

    return

end

!**********************************************************************

subroutine get_mmmm( mmmm )

    implicit none

    real*8   mmmm(124)

    mmmm(1) = 0.55D+00
    mmmm(2) = 0.6875D+00
    mmmm(3) = 0.6125D+00
    mmmm(4) = 2.24166666666667D+00
    mmmm(5) = 8D+00
    mmmm(6) = 3.97142857142857D+00
    mmmm(7) = 0.291666666666667D+00
    mmmm(8) = 0.159090909090909D+00
    mmmm(9) = 0.136363636363636D+00
    mmmm(10) = 0.241666666666667D+00
    mmmm(11) = 5.83333333333333D+00
    mmmm(12) = 0.85D+00
    mmmm(13) = 0.8D+00
    mmmm(14) = 3.44D+00
    mmmm(15) = 1.06D+00
    mmmm(16) = 0.905D+00
    mmmm(17) = 2.37142857142857D+00
    mmmm(18) = 1.9125D+00
    mmmm(19) = 1.27142857142857D+00
    mmmm(20) = 1.0625D+00
    mmmm(21) = 7.74285714285714D+00
    mmmm(22) = 7.65714285714286D+00
    mmmm(23) = 0.795D+00
    mmmm(24) = 1.76D+00
    mmmm(25) = 0.23D+00
    mmmm(26) = 1.265D+00
    mmmm(27) = 0.16D+00
    mmmm(28) = 1.59166666666667D+00
    mmmm(29) = 0.291666666666667D+00
    mmmm(30) = 2.7D+00
    mmmm(31) = 0.34D+00
    mmmm(32) = 4.35D+00
    mmmm(33) = 1.07777777777778D+00
    mmmm(34) = 0.466666666666667D+00
    mmmm(35) = 0.466666666666667D+00
    mmmm(36) = 1.15D+00
    mmmm(37) = 0.85D+00
    mmmm(38) = 2.5625D+00
    mmmm(39) = 0.7875D+00
    mmmm(40) = 2.17142857142857D+00
    mmmm(41) = 1.08571428571429D+00
    mmmm(42) = 5.17142857142857D+00
    mmmm(43) = 1.92D+00
    mmmm(44) = 0.542857142857143D+00
    mmmm(45) = 3.66666666666667D+00
    mmmm(46) = 3.45D+00
    mmmm(47) = 0.26D+00
    mmmm(48) = 0.26D+00
    mmmm(49) = 5.4D+00
    mmmm(50) = 1.7D+00
    mmmm(51) = 1.04D+00
    mmmm(52) = 0.542857142857143D+00
    mmmm(53) = 1.11666666666667D+00
    mmmm(54) = 0.383333333333333D+00
    mmmm(55) = 0.173333333333333D+00
    mmmm(56) = 0.633333333333333D+00
    mmmm(57) = 0.13D+00
    mmmm(58) = 0.9875D+00
    mmmm(59) = 0.14D+00
    mmmm(60) = 0.45D+00
    mmmm(61) = 2.13333333333333D+00
    mmmm(62) = 2.91818181818182D+00
    mmmm(63) = 2.54545454545454D+00
    mmmm(64) = 1.01111111111111D+00
    mmmm(65) = 0.27D+00
    mmmm(66) = 1.42222222222222D+00
    mmmm(67) = 0.256D+00
    mmmm(68) = 0.516666666666667D+00
    mmmm(69) = 2.61666666666667D+00
    mmmm(70) = 2.5875D+00
    mmmm(71) = 0.766666666666667D+00
    mmmm(72) = 7.17142857142857D+00
    mmmm(73) = 2.08D+00
    mmmm(74) = 1.6D+00
    mmmm(75) = 1.45D+00
    mmmm(76) = 6.68333333333333D+00
    mmmm(77) = 1.21333333333333D+00
    mmmm(78) = 1.225D+00
    mmmm(79) = 0.566666666666667D+00
    mmmm(80) = 0.293333333333333D+00
    mmmm(81) = 0.444444444444444D+00
    mmmm(82) = 0.34D+00
    mmmm(83) = 5.51764705882353D+00
    mmmm(84) = 1.85D+00
    mmmm(85) = 2.98666666666667D+00
    mmmm(86) = 3.16666666666667D+00
    mmmm(87) = 2.56551724137931D+00
    mmmm(88) = 1.63333333333333D+00
    mmmm(89) = 1.88333333333333D+00
    mmmm(90) = 7.62857142857143D+00
    mmmm(91) = 2.02222222222222D+00
    mmmm(92) = 2.51666666666667D+00
    mmmm(93) = 2.05333333333333D+00
    mmmm(94) = 0.4375D+00
    mmmm(95) = 2.16666666666667D+00
    mmmm(96) = 0.8D+00
    mmmm(97) = 1.45D+00
    mmmm(98) = 0.3D+00
    mmmm(99) = 1.54285714285714D+00
    mmmm(100) = 4.63333333333333D+00
    mmmm(101) = 0.438095238095238D+00
    mmmm(102) = 0.411111111111111D+00
    mmmm(103) = 0.6D+00
    mmmm(104) = 2.44444444444444D+00
    mmmm(105) = 0.3D+00
    mmmm(106) = 0.28D+00
    mmmm(107) = 0.21D+00
    mmmm(108) = 0.24D+00
    mmmm(109) = 0.16D+00
    mmmm(110) = 0.57D+00
    mmmm(111) = 0.28D+00
    mmmm(112) = 0.22D+00
    mmmm(113) = 0.136D+00
    mmmm(114) = 0.672D+00
    mmmm(115) = 0.988D+00
    mmmm(116) = 0.513333333333333D+00
    mmmm(117) = 0.27D+00
    mmmm(118) = 0.24D+00
    mmmm(119) = 0.3125D+00
    mmmm(120) = 0.357142857142857D+00
    mmmm(121) = 0.4825D+00
    mmmm(122) = 2.49444444444444D+00
    mmmm(123) = 2.46111111111111D+00
    mmmm(124) = 0.395454545454545D+00

    return

end

!**********************************************************************

subroutine get_aaaa( aaaa )

    implicit none

    real*8   aaaa(124)

    aaaa(1) = 0.871430143D+00
    aaaa(2) = 0.357418513D+00
    aaaa(3) = 0.357418513D+00
    aaaa(4) = 0.723048449D+00
    aaaa(5) = 0.37912345D+00
    aaaa(6) = 0.37912345D+00
    aaaa(7) = 0.529695378D+00
    aaaa(8) = 0.806790788D+00
    aaaa(9) = 0.806790788D+00
    aaaa(10) = 0.806790788D+00
    aaaa(11) = 0.606549664D+00
    aaaa(12) = 0.529695378D+00
    aaaa(13) = 0.529695378D+00
    aaaa(14) = 0.406802796D+00
    aaaa(15) = 0.529695378D+00
    aaaa(16) = 1.211460968D+00
    aaaa(17) = 0.37912345D+00
    aaaa(18) = 0.723048449D+00
    aaaa(19) = 0.723048449D+00
    aaaa(20) = 0.723048449D+00
    aaaa(21) = 0.348100645D+00
    aaaa(22) = 0.357418513D+00
    aaaa(23) = 1.110428158D+00
    aaaa(24) = 0.806790788D+00
    aaaa(25) = 0.871430143D+00
    aaaa(26) = 1.110428158D+00
    aaaa(27) = 0.723048449D+00
    aaaa(28) = 0.513833828D+00
    aaaa(29) = 0.357418513D+00
    aaaa(30) = 0.529695378D+00
    aaaa(31) = 0.529695378D+00
    aaaa(32) = 0.529695378D+00
    aaaa(33) = 0.529695378D+00
    aaaa(34) = 0.529695378D+00
    aaaa(35) = 0.357418513D+00
    aaaa(36) = 0.406802796D+00
    aaaa(37) = 0.529695378D+00
    aaaa(38) = 0.406802796D+00
    aaaa(39) = 0.806790788D+00
    aaaa(40) = 0.529695378D+00
    aaaa(41) = 0.723048449D+00
    aaaa(42) = 1.110428158D+00
    aaaa(43) = 0.723048449D+00
    aaaa(44) = 0.529695378D+00
    aaaa(45) = 0.406802796D+00
    aaaa(46) = 1.110428158D+00
    aaaa(47) = 1.211460968D+00
    aaaa(48) = 1.211460968D+00
    aaaa(49) = 1.110428158D+00
    aaaa(50) = 0.529695378D+00
    aaaa(51) = 0.723048449D+00
    aaaa(52) = 0.529695378D+00
    aaaa(53) = 0.529695378D+00
    aaaa(54) = 0.357418513D+00
    aaaa(55) = 0.871430143D+00
    aaaa(56) = 0.871430143D+00
    aaaa(57) = 1.211460968D+00
    aaaa(58) = 0.357418513D+00
    aaaa(59) = 0.357418513D+00
    aaaa(60) = 0.806790788D+00
    aaaa(61) = 0.723048449D+00
    aaaa(62) = 0.871430143D+00
    aaaa(63) = 0.806790788D+00
    aaaa(64) = 0.529695378D+00
    aaaa(65) = 0.529695378D+00
    aaaa(66) = 0.529695378D+00
    aaaa(67) = 0.529695378D+00
    aaaa(68) = 0.357418513D+00
    aaaa(69) = 0.606549664D+00
    aaaa(70) = 1.211460968D+00
    aaaa(71) = 0.606549664D+00
    aaaa(72) = 0.357418513D+00
    aaaa(73) = 0.871430143D+00
    aaaa(74) = 0.723048449D+00
    aaaa(75) = 0.723048449D+00
    aaaa(76) = 0.348100645D+00
    aaaa(77) = 0.37912345D+00
    aaaa(78) = 0.37912345D+00
    aaaa(79) = 0.529695378D+00
    aaaa(80) = 0.529695378D+00
    aaaa(81) = 0.529695378D+00
    aaaa(82) = 0.723048449D+00
    aaaa(83) = 0.529695378D+00
    aaaa(84) = 1.211460968D+00
    aaaa(85) = 0.723048449D+00
    aaaa(86) = 0.723048449D+00
    aaaa(87) = 0.723048449D+00
    aaaa(88) = 0.871430143D+00
    aaaa(89) = 0.723048449D+00
    aaaa(90) = 0.37912345D+00
    aaaa(91) = 0.529695378D+00
    aaaa(92) = 0.37912345D+00
    aaaa(93) = 0.37912345D+00
    aaaa(94) = 0.529695378D+00
    aaaa(95) = 0.37912345D+00
    aaaa(96) = 0.37912345D+00
    aaaa(97) = 0.37912345D+00
    aaaa(98) = 0.37912345D+00
    aaaa(99) = 0.723048449D+00
    aaaa(100) = 0.529695378D+00
    aaaa(101) = 0.871430143D+00
    aaaa(102) = 0.806790788D+00
    aaaa(103) = 0.406802796D+00
    aaaa(104) = 0.529695378D+00
    aaaa(105) = 0.510049917D+00
    aaaa(106) = 0.529695378D+00
    aaaa(107) = 0.529695378D+00
    aaaa(108) = 0.871430143D+00
    aaaa(109) = 0.871430143D+00
    aaaa(110) = 0.529695378D+00
    aaaa(111) = 0.529695378D+00
    aaaa(112) = 0.510049917D+00
    aaaa(113) = 0.529695378D+00
    aaaa(114) = 0.723048449D+00
    aaaa(115) = 0.723048449D+00
    aaaa(116) = 0.513833828D+00
    aaaa(117) = 0.513833828D+00
    aaaa(118) = 0.513833828D+00
    aaaa(119) = 0.513833828D+00
    aaaa(120) = 0.513833828D+00
    aaaa(121) = 0.513833828D+00
    aaaa(122) = 0.513833828D+00
    aaaa(123) = 0.513833828D+00
    aaaa(124) = 0.513833828D+00

    return

end

!**********************************************************************

subroutine get_ssss( ssss )

    implicit none

    integer  ssss(68)

    ssss(1) = 1
    ssss(2) = 1
    ssss(3) = 1
    ssss(4) = 1
    ssss(5) = 1
    ssss(6) = 1
    ssss(7) = -1
    ssss(8) = 1
    ssss(9) = 1
    ssss(10) = -1
    ssss(11) = -1
    ssss(12) = -1
    ssss(13) = -1
    ssss(14) = -1
    ssss(15) = -1
    ssss(16) = -1
    ssss(17) = -1
    ssss(18) = -1
    ssss(19) = -1
    ssss(20) = -1
    ssss(21) = -1
    ssss(22) = 1
    ssss(23) = 1
    ssss(24) = 1
    ssss(25) = 1
    ssss(26) = 1
    ssss(27) = 1
    ssss(28) = 1
    ssss(29) = 1
    ssss(30) = 1
    ssss(31) = 1
    ssss(32) = 1
    ssss(33) = 1
    ssss(34) = 1
    ssss(35) = 1
    ssss(36) = 1
    ssss(37) = 1
    ssss(38) = 1
    ssss(39) = 1
    ssss(40) = 1
    ssss(41) = 1
    ssss(42) = 1
    ssss(43) = 1
    ssss(44) = 1
    ssss(45) = 1
    ssss(46) = 1
    ssss(47) = 1
    ssss(48) = 1
    ssss(49) = 1
    ssss(50) = 1
    ssss(51) = 1
    ssss(52) = 1
    ssss(53) = 1
    ssss(54) = 1
    ssss(55) = 1
    ssss(56) = 1
    ssss(57) = 1
    ssss(58) = 1
    ssss(59) = 1
    ssss(60) = 1
    ssss(61) = 1
    ssss(62) = 1
    ssss(63) = 1
    ssss(64) = 1
    ssss(65) = 1
    ssss(66) = 1
    ssss(67) = 1
    ssss(68) = 1

    return

end


!*************************************************************
    
subroutine get_bbbb(xl, xu, gu)

    implicit none

    real*8 xl(124), xu(124), gu(68)

    xl(1) = 0.7D+00
    xl(2) = 0.7D+00
    xl(3) = 0.7D+00
    xl(4) = 0.7D+00
    xl(5) = 0.7D+00
    xl(6) = 0.7D+00
    xl(7) = 0.7D+00
    xl(8) = 0.7D+00
    xl(9) = 0.7D+00
    xl(10) = 0.7D+00
    xl(11) = 0.7D+00
    xl(12) = 0.7D+00
    xl(13) = 0.7D+00
    xl(14) = 0.7D+00
    xl(15) = 0.7D+00
    xl(16) = 0.7D+00
    xl(17) = 0.7D+00
    xl(18) = 0.7D+00
    xl(19) = 0.7D+00
    xl(20) = 0.7D+00
    xl(21) = 0.7D+00
    xl(22) = 0.7D+00
    xl(23) = 0.7D+00
    xl(24) = 0.7D+00
    xl(25) = 0.7D+00
    xl(26) = 0.7D+00
    xl(27) = 0.7D+00
    xl(28) = 0.7D+00
    xl(29) = 0.7D+00
    xl(30) = 0.7D+00
    xl(31) = 0.7D+00
    xl(32) = 0.7D+00
    xl(33) = 0.7D+00
    xl(34) = 0.7D+00
    xl(35) = 0.7D+00
    xl(36) = 0.7D+00
    xl(37) = 0.7D+00
    xl(38) = 0.7D+00
    xl(39) = 0.7D+00
    xl(40) = 0.7D+00
    xl(41) = 0.7D+00
    xl(42) = 0.7D+00
    xl(43) = 0.7D+00
    xl(44) = 0.7D+00
    xl(45) = 0.7D+00
    xl(46) = 0.7D+00
    xl(47) = 0.7D+00
    xl(48) = 0.7D+00
    xl(49) = 0.7D+00
    xl(50) = 0.7D+00
    xl(51) = 0.7D+00
    xl(52) = 0.7D+00
    xl(53) = 0.7D+00
    xl(54) = 0.7D+00
    xl(55) = 0.7D+00
    xl(56) = 0.7D+00
    xl(57) = 0.7D+00
    xl(58) = 0.7D+00
    xl(59) = 0.7D+00
    xl(60) = 0.7D+00
    xl(61) = 0.7D+00
    xl(62) = 0.7D+00
    xl(63) = 0.7D+00
    xl(64) = 0.7D+00
    xl(65) = 0.7D+00
    xl(66) = 0.7D+00
    xl(67) = 0.7D+00
    xl(68) = 0.7D+00
    xl(69) = 0.7D+00
    xl(70) = 0.7D+00
    xl(71) = 0.7D+00
    xl(72) = 0.7D+00
    xl(73) = 0.7D+00
    xl(74) = 0.7D+00
    xl(75) = 0.7D+00
    xl(76) = 0.7D+00
    xl(77) = 0.7D+00
    xl(78) = 0.7D+00
    xl(79) = 0.7D+00
    xl(80) = 0.7D+00
    xl(81) = 0.7D+00
    xl(82) = 0.7D+00
    xl(83) = 0.7D+00
    xl(84) = 0.7D+00
    xl(85) = 0.7D+00
    xl(86) = 0.7D+00
    xl(87) = 0.7D+00
    xl(88) = 0.7D+00
    xl(89) = 0.7D+00
    xl(90) = 0.7D+00
    xl(91) = 0.7D+00
    xl(92) = 0.7D+00
    xl(93) = 0.7D+00
    xl(94) = 0.7D+00
    xl(95) = 0.7D+00
    xl(96) = 0.7D+00
    xl(97) = 0.7D+00
    xl(98) = 0.7D+00
    xl(99) = 0.7D+00
    xl(100) = 0.7D+00
    xl(101) = 0.7D+00
    xl(102) = 0.7D+00
    xl(103) = 0.7D+00
    xl(104) = 0.7D+00
    xl(105) = 0.7D+00
    xl(106) = 0.7D+00
    xl(107) = 0.7D+00
    xl(108) = 0.7D+00
    xl(109) = 0.7D+00
    xl(110) = 0.7D+00
    xl(111) = 0.7D+00
    xl(112) = 0.7D+00
    xl(113) = 0.7D+00
    xl(114) = 1.5D+00
    xl(115) = 1.5D+00
    xl(116) = 1.5D+00
    xl(117) = 1.5D+00
    xl(118) = 2.5D+00
    xl(119) = 2.5D+00
    xl(120) = 2.5D+00
    xl(121) = 2.5D+00
    xl(122) = 1.5D+00
    xl(123) = 1.5D+00
    xl(124) = 1.5D+00

    xu(1) = 2.2D+00
    xu(2) = 2.2D+00
    xu(3) = 2.2D+00
    xu(4) = 2.2D+00
    xu(5) = 2.2D+00
    xu(6) = 2.2D+00
    xu(7) = 2.2D+00
    xu(8) = 2.2D+00
    xu(9) = 2.2D+00
    xu(10) = 2.4D+00
    xu(11) = 2.2D+00
    xu(12) = 2.2D+00
    xu(13) = 2.2D+00
    xu(14) = 2.2D+00
    xu(15) = 2.2D+00
    xu(16) = 2.2D+00
    xu(17) = 2.2D+00
    xu(18) = 2.2D+00
    xu(19) = 2.2D+00
    xu(20) = 2.2D+00
    xu(21) = 2.2D+00
    xu(22) = 2.2D+00
    xu(23) = 2.2D+00
    xu(24) = 2.2D+00
    xu(25) = 2.2D+00
    xu(26) = 2.2D+00
    xu(27) = 2.2D+00
    xu(28) = 2.2D+00
    xu(29) = 2.2D+00
    xu(30) = 2.2D+00
    xu(31) = 2.2D+00
    xu(32) = 2.2D+00
    xu(33) = 2.2D+00
    xu(34) = 2.2D+00
    xu(35) = 2.2D+00
    xu(36) = 2.2D+00
    xu(37) = 2.2D+00
    xu(38) = 2.2D+00
    xu(39) = 2.2D+00
    xu(40) = 2.2D+00
    xu(41) = 2.2D+00
    xu(42) = 2.2D+00
    xu(43) = 2.2D+00
    xu(44) = 2.2D+00
    xu(45) = 2.2D+00
    xu(46) = 2.2D+00
    xu(47) = 2.2D+00
    xu(48) = 2.2D+00
    xu(49) = 2.2D+00
    xu(50) = 2.2D+00
    xu(51) = 2.2D+00
    xu(52) = 2.2D+00
    xu(53) = 2.2D+00
    xu(54) = 2.2D+00
    xu(55) = 2.2D+00
    xu(56) = 2.2D+00
    xu(57) = 2.2D+00
    xu(58) = 2.2D+00
    xu(59) = 2.2D+00
    xu(60) = 2.2D+00
    xu(61) = 2.2D+00
    xu(62) = 2.2D+00
    xu(63) = 2.2D+00
    xu(64) = 2.2D+00
    xu(65) = 2.2D+00
    xu(66) = 2.2D+00
    xu(67) = 2.5D+00
    xu(68) = 2.2D+00
    xu(69) = 2.2D+00
    xu(70) = 2.2D+00
    xu(71) = 2.2D+00
    xu(72) = 2.2D+00
    xu(73) = 2.2D+00
    xu(74) = 2.2D+00
    xu(75) = 2.2D+00
    xu(76) = 2.2D+00
    xu(77) = 2.2D+00
    xu(78) = 2.2D+00
    xu(79) = 2.2D+00
    xu(80) = 2.2D+00
    xu(81) = 2.2D+00
    xu(82) = 2.2D+00
    xu(83) = 2.2D+00
    xu(84) = 2.2D+00
    xu(85) = 2.2D+00
    xu(86) = 2.2D+00
    xu(87) = 2.2D+00
    xu(88) = 2.2D+00
    xu(89) = 2.2D+00
    xu(90) = 2.2D+00
    xu(91) = 2.2D+00
    xu(92) = 2.2D+00
    xu(93) = 2.2D+00
    xu(94) = 2.2D+00
    xu(95) = 2.2D+00
    xu(96) = 2.2D+00
    xu(97) = 2.2D+00
    xu(98) = 2.2D+00
    xu(99) = 2.2D+00
    xu(100) = 2.2D+00
    xu(101) = 2.2D+00
    xu(102) = 2.2D+00
    xu(103) = 2.2D+00
    xu(104) = 2.2D+00
    xu(105) = 2.2D+00
    xu(106) = 2.2D+00
    xu(107) = 2.2D+00
    xu(108) = 2.2D+00
    xu(109) = 2.2D+00
    xu(110) = 2.2D+00
    xu(111) = 2.2D+00
    xu(112) = 2.2D+00
    xu(113) = 2.5D+00
    xu(114) = 3D+00
    xu(115) = 3D+00
    xu(116) = 3.5D+00
    xu(117) = 3.5D+00
    xu(118) = 4.2D+00
    xu(119) = 4.2D+00
    xu(120) = 4.2D+00
    xu(121) = 4.2D+00
    xu(122) = 2D+00
    xu(123) = 2D+00
    xu(124) = 2.5D+00

    gu(1) = 27D+00
    gu(2) = 95D+00
    gu(3) = 95D+00
    gu(4) = 95D+00
    gu(5) = 28D+00
    gu(6) = 30D+00
    gu(7) = -70D+00
    gu(8) = 70D+00
    gu(9) = 10D+00
    gu(10) = -14D+00
    gu(11) = -10D+00
    gu(12) = -10D+00
    gu(13) = -10D+00
    gu(14) = -10D+00
    gu(15) = -33.39D+00
    gu(16) = -36.7199999999999D+00
    gu(17) = -6D+00
    gu(18) = -5D+00
    gu(19) = -35D+00
    gu(20) = -43.4799999999999D+00
    gu(21) = -40D+00
    gu(22) = 394.778349999999D+00
    gu(23) = 180D+00
    gu(24) = 538.24323D+00
    gu(25) = 1550.2844D+00
    gu(26) = 1231.50739999999D+00
    gu(27) = 2859.9976D+00
    gu(28) = 180.997729999999D+00
    gu(29) = 180D+00
    gu(30) = 273.58575D+00
    gu(31) = 493.07184D+00
    gu(32) = 309.93292D+00
    gu(33) = 242.52582D+00
    gu(34) = 313.9902D+00
    gu(35) = 230.407939999999D+00
    gu(36) = 370.60809D+00
    gu(37) = 180D+00
    gu(38) = 261.613069999999D+00
    gu(39) = 180D+00
    gu(40) = 525.57D+00
    gu(41) = 180D+00
    gu(42) = 14812.8009999999D+00
    gu(43) = 474.62454D+00
    gu(44) = 14658.682D+00
    gu(45) = 486.276729999999D+00
    gu(46) = 430.6485D+00
    gu(47) = 180D+00
    gu(48) = 246.376139999999D+00
    gu(49) = 180D+00
    gu(50) = 368.63852D+00
    gu(51) = 180D+00
    gu(52) = 469.37457D+00
    gu(53) = 180D+00
    gu(54) = 289.55148D+00
    gu(55) = 180D+00
    gu(56) = 75D+00
    gu(57) = 75D+00
    gu(58) = 75D+00
    gu(59) = 70D+00
    gu(60) = 70D+00
    gu(61) = 70D+00
    gu(62) = 70D+00
    gu(63) = 70D+00
    gu(64) = 30D+00
    gu(65) = 30D+00
    gu(66) = 20D+00
    gu(67) = 20D+00
    gu(68) = 20D+00

    return

end
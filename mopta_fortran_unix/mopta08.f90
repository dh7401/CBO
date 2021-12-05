
program mopta08

    implicit      none

    character*200  in_file
    character*200  out_file
    integer        nvar
    integer        ncon
    parameter      (nvar = 124)
    parameter      (ncon =  68)
    integer        i
    real*8         x(nvar)
    real*8         f
    real*8         g(ncon)
    
    in_file  = "input.txt"
    out_file = "output.txt"

    open(7,FILE=in_file)
    do i=1,nvar
        read(7,*) x(i)
    enddo
    close(7)

    call func(nvar,ncon,x,f,g)

    open(8,FILE=out_file)
    write(8,'(F27.16)') f
    do i=1,ncon
        write(8,'(F27.16)') g(i)
    enddo
    close(8)

    write(*,'(a)') 'done'

    stop

end

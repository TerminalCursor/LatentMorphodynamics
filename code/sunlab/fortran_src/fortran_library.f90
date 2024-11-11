! fortran_library

! Mean-Squared Displacement for a time lag `t`
function msd(a, t) result(b)
    implicit none
    
    real(kind=8), intent(in) :: a(:,:)
    integer(kind=8), intent(in) :: t
    integer(kind=8) :: i, j
    real(kind=8) :: s(size(a,1)-t, size(a,2))
    real(kind=8) :: b(size(a,1)-t)

    s(:, :) = a((t + 1):(size(a,1)), :) - a(1:(size(a,1)-t),:)
    do i = 1, size(a, 1)-t
        b(i) = 0.0
        do j = 1, size(a, 2)
            b(i) = b(i) + s(i, j) ** 2
        end do
        b(i) = sqrt(b(i))
    end do
end function msd

! Mean-Squared Displacement for a time lag `t`
function mmsd(a, t) result(z)
    implicit none
    
    real(kind=8), intent(in) :: a(:,:)
    integer(kind=8), intent(in) :: t
    integer(kind=8) :: i, j
    real(kind=8) :: s(size(a,1)-t, size(a,2))
    real(kind=8) :: b(size(a,1)-t)
    real(kind=8) :: z

    s(:, :) = a((t + 1):(size(a,1)), :) - a(1:(size(a,1)-t),:)
    do i = 1, size(a, 1)-t
        b(i) = 0.0
        do j = 1, size(a, 2)
            b(i) = b(i) + s(i, j) ** 2
        end do
        b(i) = sqrt(b(i))
    end do
    z = sum(b(:)) / (size(b, 1))
end function mmsd

! Mean-Squared Displacement for time lags up to `T`
function msds(a, Ts) result(b)
    implicit none
    
    real(kind=8), intent(in) :: a(:,:)
    integer(kind=8), intent(in) :: Ts
    integer(kind=8) :: i, j
    integer(kind=8) :: t
    real(kind=8) :: s(size(a,1), size(a,2))
    real(kind=8) :: b(size(a,1),Ts+1)

    do t = 0, Ts
        s(:, :) = a((t + 1):(size(a,1)), :) - a(1:(size(a,1)-t),:)
        do i = 1, size(a, 1)
            b(i,t+1) = 0.0
            if (i <= size(a, 1) - t) then
                do j = 1, size(a, 2)
                    b(i,t+1) = b(i,t+1) + s(i, j) ** 2
                end do
                b(i,t+1) = sqrt(b(i,t+1))
            end if
        end do
    end do
end function msds

! Mean-Squared Displacement for time lags up to `T`
! Return the mean for each time lag
function mmsds(a, Ts) result(z)
    implicit none
    
    real(kind=8), intent(in) :: a(:,:)
    integer(kind=8), intent(in) :: Ts
    integer(kind=8) :: i, j
    integer(kind=8) :: t
    real(kind=8) :: s(size(a,1), size(a,2))
    real(kind=8) :: b(size(a,1),Ts+1)
    real(kind=8) :: z(Ts+1)

    do t = 0, Ts
        s(:, :) = a((t + 1):(size(a,1)), :) - a(1:(size(a,1)-t),:)
        do i = 1, size(a, 1)
            b(i,t+1) = 0.0
            if (i <= size(a, 1) - t) then
                do j = 1, size(a, 2)
                    b(i,t+1) = b(i,t+1) + s(i, j) ** 2
                end do
                b(i,t+1) = sqrt(b(i,t+1))
            end if
        end do
        z(t+1) = sum(b(1:(size(a, 1) - t),t+1)) / (size(a, 1) - t)
    end do
end function mmsds

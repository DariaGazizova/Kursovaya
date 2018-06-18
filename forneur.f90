program neurower

! initialize I-init layer, H-hidden layer, O-out layer, w0-weight between init and hidden layer, w1-weights between hidden and out layer, y -result
   real,allocatable :: I(:,:),H(:,:),O(:),w0(:,:),w1(:,:),per(:,:),reading(:,:),per1(:),y(:)
   real,allocatable :: O_error(:),O_delta(:),H_error(:,:),H_delta(:,:)
   real :: Hid, comb, init, rand, deriv,error
   integer :: z,j,l,m
!OMP directives
   integer :: num_threads,  OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
   double precision :: time ,OMP_GET_WTIME
   call init_random_seed(seed)
   Hid=1000 !number of hidden neurons
   comb=1000 ! number of combinations 
   init=10 !number of init neurons
   rand=20
   allocate(I(comb,init),H(comb,Hid),O(comb),w0(init,Hid),w1(Hid,1))
   allocate(per(comb,Hid),reading(1,20),per1(comb),y(comb))
   allocate(O_error(comb),O_delta(comb),H_error(comb,Hid),H_delta(comb,Hid))
   open(1, file="init.txt")
   open(2, file="result.txt")
   do z=1,comb
      read(1, *)reading(1,:)
      do j=1,init
         I(z,init+1-j)=reading(1,21-j)
      end do
      read(2, *)y(z)
   end do
!initialize hidden and out layer
   H=0
   O=0
   per=0
   per1=0
   call random_number(w0)
   call random_number(w1)
!randomly initialize weights
   w0=rand*(w0-0.5)
   w1=rand*(w1-0.5)
   time=OMP_GET_WTIME()    
   do z=1,5000
      error=0
      per=0
!find hidden layer
      do j=1,comb
         do l=1,Hid
            do m=1,init
               per(j,l)=per(j,l)+I(j,m)*w0(m,l)
            end do
         end do
      end do
      !$OMP PARALLEL DO
      do m=1,comb
         do j=1,Hid      
            H(m,j)=1/(1+exp(-per(m,j)))
         end do
      end do
      !$OMP END PARALLEL DO
!find out layer
      per1=0
      do j=1,comb
         do m=1,Hid
               per1(j)=per1(j)+H(j,m)*w1(m,1)
         end do
      end do
      !$OMP PARALLEL DO
      do m=1,comb     
            O(m)=1/(1+exp(-per1(m)))
      end do
      !$OMP END PARALLEL DO
!find out error
      O_error=y-O
      do j=1,comb
      end do
      do j=1,comb
         error=error+abs(O_error(j))
      end do
      print *,error/comb,z 
!find out delta  
      do j=1,comb
         O_delta(j)=O_error(j)*O(j)*(1-O(j))
      end do
!find hidden layer error
      do j=1,comb
         do m=1,Hid
            H_error(j,m)=O_delta(j)*w1(m,1)
         end do
      end do
!find hidden layer delta
      do j=1,comb
         do m=1,Hid
            H_delta(j,m)=H_error(j,m)*H(j,m)*(1-H(j,m))
         end do
      end do
!find new weights
      H=transpose(H)
      do j=1,Hid
         do m=1,comb
            w1(j,1)=w1(j,1)+H(j,m)*O_delta(m)
         end do
      end do
      H=transpose(H)
      I=transpose(I)
      do j=1,init
         do l=1,Hid
            do m=1,comb
            end do
               w0(j,l)=w0(j,l)+I(j,m)*H_delta(m,l)
         end do
      end do
      I=transpose(I)
   end do
   do z=1,comb
   print *,O(z),y(z)
   end do
   time=OMP_GET_WTIME()-time
   print *,  time
end


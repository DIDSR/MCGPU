
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                      C
C   This program reads a PENELOPE 2006 material file and outputs a     C
C   table with photon interaction mean free paths (MFP), and data for  C
C   Rayleigh and Compton interaction sampling.                         C
C                                                                      C
C   While the PENELOPE database is linearly interpolated in LOG-LOG,   C
C   the energy grid in the output table is a linear, ie, has equally   C
C   spaced energy bins. A small bin width is required to allow direct  C
C   linear interpolation of the MFP, avoiding the LOG computation.     C
C                                                                      C
C   This source code is based on PENELOPE's "tables.f".                C
C                                                                      C 
C                              Andreu Badal, 2009-03-31                C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C --Copyright notice from tables.f and penelope.f:                     C
C                                                                      C
C  PENELOPE/PENGEOM (version 2006)                                     C
C  Copyright (c) 2001-2006                                             C
C  Universitat de Barcelona                                            C
C                                                                      C
C  Permission to use, copy, modify, distribute and sell this software  C
C  and its documentation for any purpose is hereby granted without     C
C  fee, provided that the above copyright notice appears in all        C
C  copies and that both that copyright notice and this permission      C
C  notice appear in all supporting documentation. The Universitat de   C
C  Barcelona makes no representations about the suitability of this    C
C  software for any purpose. It is provided 'as is' without express    C
C  or implied warranty.                                                C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


C  *********************************************************************
C                       MAIN PROGRAM
C  *********************************************************************
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER*80 MFNAME, OUTNAME
      CHARACTER*62 NAME
      
C  ****  Auxiliary arrays.
      DIMENSION E_MFP(6)
      PARAMETER (MAX_ENERGY_BINS=60005)
      DIMENSION PMAX_linear_energy(MAX_ENERGY_BINS)
            
C  ****  Simulation parameters.
      PARAMETER (MAXMAT=10)
      COMMON/CSIMPA/EABS(3,MAXMAT),C1(MAXMAT),C2(MAXMAT),WCC(MAXMAT),
     1  WCR(MAXMAT)
C  ****  Composition data.
      COMMON/COMPOS/STF(MAXMAT,30),ZT(MAXMAT),AT(MAXMAT),RHO(MAXMAT),
     1  VMOL(MAXMAT),IZ(MAXMAT,30),NELEM(MAXMAT)

C  ****  Penelope energy grid and Rayleigh sampling data:
      PARAMETER (NEGP=200)
      PARAMETER (NP=128,NPM1=NP-1)
      COMMON/CGRA/XCO(NP,MAXMAT),PCO(NP,MAXMAT),ACO(NP,MAXMAT),
     1  BCO(NP,MAXMAT),PMAX(NEGP,MAXMAT),ITLCO(NP,MAXMAT),
     2  ITUCO(NP,MAXMAT)

C  ****  Energy grid and interpolation constants for the current energy.
      COMMON/CEGRID/EL,EU,ET(NEGP),DLEMP(NEGP),DLEMP1,DLFC,
     1  XEL,XE,XEK,KE

C  ****  Compton scattering.
      PARAMETER (NOCO=64)
      COMMON/CGCO/FCO(MAXMAT,NOCO),UICO(MAXMAT,NOCO),FJ0(MAXMAT,NOCO),
     2  KZCO(MAXMAT,NOCO),KSCO(MAXMAT,NOCO),NOSCCO(MAXMAT)     



      WRITE(6,*)" "
      WRITE(6,*)" "
      WRITE(6,*)"         ***********************************"
      WRITE(6,*)"         *** MC-GPU_create_material_data ***"
      WRITE(6,*)"         ***********************************"
      WRITE(6,*)" "
      WRITE(6,*)" "
      WRITE(6,*)"    Creating a material input file for MC-GPU."
      WRITE(6,*)
     &"    This program reads a PENELOPE 2006 material file and outputs"
      WRITE(6,*)
     & "    a table with photon interaction mean free paths (MFP) and"
      WRITE(6,*)
     & "    data for Rayleigh and Compton interaction sampling."
C
C  ****  Parameters (to tabulate the complete energy range and to switch
C        soft interactions off).
C

C
C  ****  Material data file.
C
      WRITE(6,'(a)') '  '
      WRITE(6,'(a)') '  -- Enter the energy range to tabulate: '//
     &           ' Emin, Emax (eg, 5000  125000):'
      READ(5,*) EMIN, EMAX
      WRITE(6,'(a)') '  -- Enter the number of energy bins (eg, 8192):'    ! 8192 = 2^13
      READ(5,*) NBINS
      DE=(EMAX-EMIN)/DBLE(NBINS)
      WRITE(6,'(a,1pe17.10)')
     &      '      - Energy bin width set to (EMAX-EMIN)/NBINS = ',DE
      WRITE(6,'(a)') '  -- Enter the name of the PENELOPE 2006'//
     &               ' material data file (eg, water.mat):'
      READ(5,'(A80)') MFNAME
      WRITE(6,'(a)') '  -- Enter the name of the output data file'//
     &               'for MC-GPU (eg, water.mcgpu)...'
      READ(5,'(A80)') OUTNAME
      WRITE(6,'('' Material data file:  '', A40)') MFNAME
      WRITE(6,'(a)') '  '
      WRITE(6,'(a)') 'Processing material data. Please, wait...'


      ! -- Initializing PENELOPE with the material information:
      !    Tabulate the material tables between the input maximum and minimum energies.
      DO M=1,MAXMAT
        EABS(1,M) = EMIN
        EABS(2,M) = EMIN
        EABS(3,M) = EMIN
        C1(M)     =  0.0D0
        C2(M)     =  0.0D0
        WCC(M)    =  0.0D0
        WCR(M)    =-10.0D0
      ENDDO

      OPEN(11,FILE=MFNAME)
      CALL PEINIT(EMAX,1,11,6,1)    !! Last parameter controls the amount of info output. Use '5' for maximum info.
      CLOSE(11)

      ! -- Re-open the material file and read the material name (2nd line):
      OPEN(11,FILE=MFNAME)
      READ(11,'(A55)') NAME
      READ(11,'(11X,a62)') NAME
      CLOSE(11)
 

C  ****  Calculate photon mean free paths:
C    ** Function PHMFP returns the mean free path,MFP, [cm] for the input energy, kind of particle,
C    **  material number (from input file), and kind of interaction.                         
C    ** The cross section is found dividing the inverse MFP by the molar volume [atoms/cm^3].
C
      WRITE(6,*)'  '
c      WRITE(6,*)'====================================================='
c      write(6,*)' PENELOPEs function PHMFP returns the mean free'//
c     &          ' path [cm] for the input energy, kind of particle, '//
c     &          ' material number, and kind of interaction: '
c      write(6,*)'    MFP=PHMFP(E,KPAR,M,ICOL)'
c      write(6,*)' The cross section is found dividing the inverse'//
c     &          ' mean free path (=attenuation coefficient) by the'//
c     &          ' molar volume [atoms/cm^3].'
c      write(6,*)'    XS=(1.0D0/MFP)/VMOL(M)'
c     WRITE(6,*)'====================================================='
c      WRITE(6,*) '  '      

      ! Set mat number and particle:
      M=1                ! Use first material defined in the input material file
      KPAR = 2           ! Select photons (1=electron, 2=photon, 3=positron)
      
      ! -- Open output file:
      OPEN(1, FILE=OUTNAME)

      ! -- Write file header:
      WRITE(1,'(a)')'#[MATERIAL DEFINITION FOR MC-GPU: interaction'//
     &      ' mean free path and sampling data from PENELOPE 2006]'
      WRITE(1,'(a)')'#[MATERIAL NAME]'
      WRITE(1,1001) NAME
 1001 format('# ',a)
      WRITE(1,'(a)')'#[NOMINAL DENSITY (g/cm^3)]'
      WRITE(1,1002) RHO(M)
 1002 format('# ',f12.8)
      WRITE(1,'(a)')'#[NUMBER OF DATA VALUES]'
      WRITE(1,1003) NBINS
 1003 format('# ',I6)
      WRITE(1,'(a)')'#[MEAN FREE PATHS (cm)'//
     &        ' (ie, average distance between interactions)]'
      WRITE(1,'(a)') '#[Energy (eV)     | Rayleigh        |'//
     &                ' Compton         | Photoelectric   |'//           
     &                ' TOTAL (+pair prod) (cm) |'//            !  &  ' Pair-production | TOTAL (cm) |'//
     &                ' Rayleigh: max cumul prob F^2]'


ccccc *** MEAN FREE PATH DATA (and Rayleigh cumulative prob) **********

      ! -- Re-calculate the maximum Rayleigh cumulative probability for each linear energy bin instead of the PENELOPE grid:
      call GRAaI_linear_energy(M, NBINS, EMIN, DE, PMAX_linear_energy)


      do i = 1, NBINS

        E = EMIN + (i-1)*DE             ! Set bin energy

        IF(E.LT.EABS(KPAR,M).OR.E.GT.EMAX) THEN
          WRITE(6,*) '!!ERROR!! Energy outside the table interval!',
     &               ' #bin, E = ', i, E
          STOP 'ERROR!'
        ENDIF

        E_MFP(1) = E                    ! Store the bin energy
        E_MFP(2) = PHMFP(E,KPAR,M,1)    ! Store the bin MFPs: (1) Rayleigh
        E_MFP(3) = PHMFP(E,KPAR,M,2)    ! Store the bin MFPs: (2) Compton
        E_MFP(4) = PHMFP(E,KPAR,M,3)    ! Store the bin MFPs: (3) photoelectric
        E_MFP(5) = PHMFP(E,KPAR,M,4)    ! Store the bin MFPs: (4) pair production

        E_MFP(6) = 1.0/E_MFP(2)+1.0/E_MFP(3)+1.0/E_MFP(4)+1.0/E_MFP(5)
        E_MFP(6) = 1.0/E_MFP(6)         ! Store the bin total MFP

        write(1,'(6(1x,1pe17.10))') E_MFP(1), E_MFP(2), E_MFP(3),    ! Write MFP table to external file
     &                 E_MFP(4), E_MFP(6), PMAX_linear_energy(i)     ! Write the Rayleigh cumulative probability for the energy bin

          ! E_MFP(5) --> Pair production MFP is not written bc it is not used in the simulation, but it is included in the TOTAL MFP
     

      enddo


ccccc *** RAYLEIGH DATA ***********************************************

      ! -- Rayleigh sampling data header:
      WRITE(1,'(a)')'#[RAYLEIGH INTERACTIONS (RITA sampling '//
     &              ' of atomic form factor from EPDL database)]'
      WRITE(1,'(a)')
     & '#[DATA VALUES TO SAMPLE SQUARED MOLECULAR FORM FACTOR (F^2)]'
      WRITE(1,1003) NP
      WRITE(1,'(a)')
     & '#[SAMPLING DATA FROM COMMON/CGRA/: X, P, A, B, ITL, ITU]'   ! X == momentum transfer data value (adaptive grid), tabulated from the minimum to the maximum possible momentum transfers
                                                                     ! P == squared Molecular Form Factor cumulative prob at this X (adaptive grid)
                                                                     ! A & B == RITA sampling parameters
                                                                     ! ITL & ITU == lower and upper limits to speed binary search
      do i = 1, NP
        write(1,5555) XCO(i,M), PCO(i,M), ACO(i,M),
     1                BCO(i,M), ITLCO(i,M), ITUCO(i,M)
      enddo
5555  format(4(1x,1pe17.10),1x,i4,1x,i4)


ccccc *** COMPTON DATA ************************************************
 
      ! -- Compton sampling data header:
      WRITE(1,'(a)')
     &  '#[COMPTON INTERACTIONS (relativistic impulse model with'//
     &  ' approximated one-electron analytical profiles)]'
      WRITE(1,'(a)')'#[NUMBER OF SHELLS]'
      WRITE(1,1003) NOSCCO(M)
      WRITE(1,'(a)')'#[SHELL INFORMATION FROM COMMON/CGCO/:'//      ! FCO == equivalent number of electrons in the shell?? (eq. 2.36 penelope 2008)
     &              ' FCO, UICO, FJ0, KZCO, KSCO]'                   ! UICO == shell ionization energy
                                                                     ! FJ0 == one-electron shell profile at p_z=0 (eq. 2.54, page 72, penelope 2008)
                                                                     ! KZCO == element that "owns" the shell??
                                                                     ! KSCO == atomic shell number, ie, atomic transition line
                                                                     ! NOSCCO == number of shells, after grouping 
      do i = 1, NOSCCO(M)
        write(1,5107) FCO(M,i), UICO(M,i), FJ0(M,i),
     &                KZCO(M,i), KSCO(M,i)
      enddo
 5107 format(3(1X,E16.8),2(1X,I4))


      WRITE(1,'(a)')' '
      CLOSE(1)

      WRITE(6,'(a)')
     & '*** Material file correctly generated. Have a nice simulation!'
      WRITE(6,*)' '
     
      END




C  *********************************************************************
C        Code based on PENELOPE's subroutine:  SUBROUTINE GRAaI
C  *********************************************************************
      SUBROUTINE GRAaI_linear_energy(M, nbins, emin, de, PMAX_linear_e)
C
C  Re-init random sampling for Rayleigh scattering using the input linear energy scale
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER*2 LASYMB
      PARAMETER (REV=5.10998918D5)  ! Electron rest energy (eV)
      PARAMETER (RREV=1.0D0/REV)
C  ****  Composition data.
      PARAMETER (MAXMAT=10)
      COMMON/COMPOS/STF(MAXMAT,30),ZT(MAXMAT),AT(MAXMAT),RHO(MAXMAT),
     1  VMOL(MAXMAT),IZ(MAXMAT,30),NELEM(MAXMAT)
C  ****  Element data.
      COMMON/CADATA/ATW(99),EPX(99),RA1(99),RA2(99),RA3(99),RA4(99),
     1  RA5(99),RSCR(99),ETA(99),EB(99,30),IFI(99,30),IKS(99,30),
     2  NSHT(99),LASYMB(99)
C  ****  Energy grid and interpolation constants for the current energy.
      PARAMETER (NEGP=200)
      COMMON/CEGRID/EL,EU,ET(NEGP),DLEMP(NEGP),DLEMP1,DLFC,
     1  XEL,XE,XEK,KE
C
      PARAMETER (NM=512)
      COMMON/CRITA/XTI(NM),PACI(NM),AI(NM),BI(NM),NPI,
     1             ITTLI(NM),ITTUI(NM),NPM1I
C
      PARAMETER (NP=128)
c      COMMON/CGRA/XCO(NP,MAXMAT),PCO(NP,MAXMAT),ACO(NP,MAXMAT),
c     1  BCO(NP,MAXMAT),PMAX(NEGP,MAXMAT),ITLCO(NP,MAXMAT),
c     2  ITUCO(NP,MAXMAT)
C
      PARAMETER (NIP=51)
      DIMENSION XI(NIP),FUN(NIP),SUM(NIP)
      COMMON/CGRA00/FACTE,X2MAX,MM,MOM
      EXTERNAL GRAaD1

      !! Dimension output array:
      PARAMETER (MAX_ENERGY_BINS=50000)
      DIMENSION PMAX_linear_e(MAX_ENERGY_BINS)
      
C
      IZZ=0
      DO I=1,NELEM(M)
        IZZ=MAX(IZZ,IZ(M,I))
      ENDDO
C
      MM=M
      X2MIN=0.0D0
      X2MAX=4.0D0*20.6074D0**2*(200.0D0*IZZ)**2
      NPT=NP
      NU=NPT/4
      CALL RITAI0(GRAaD1,X2MIN,X2MAX,NPT,NU,ERRM,0)
C
C  ****  Upper limit of the X2 interval for the PENELOPE grid energies.
C
      ! OLD code: DO IE=1,NEGP
      do IE = 1, nbins
        ! OLD code: XM=2.0D0*20.6074D0*ET(IE)*RREV      !! ET(IE) is the minimum bin energy in PENELOPE's grid
        XM=2.0D0*20.6074D0*(emin+(IE-1)*de)*RREV        !! re-calculating energy with the linear scale           !!DeBuG!!
        
        X2M=XM*XM
        IF(X2M.GT.XTI(1)) THEN
          IF(X2M.LT.XTI(NP)) THEN
            I=1
            J=NPI
    1       IT=(I+J)/2
            IF(X2M.GT.XTI(IT)) THEN
              I=IT
            ELSE
              J=IT
            ENDIF
            IF(J-I.GT.1) GO TO 1
C
            X1=XTI(I)
            X2=X2M
            DX=(X2-X1)/DBLE(NIP-1)
            DO K=1,NIP
              XI(K)=X1+DBLE(K-1)*DX
              TAU=(XI(K)-XTI(I))/(XTI(I+1)-XTI(I))
              CON1=2.0D0*BI(I)*TAU
              CI=1.0D0+AI(I)+BI(I)
              CON2=CI-AI(I)*TAU
              IF(ABS(CON1).GT.1.0D-16*ABS(CON2)) THEN
                ETAP=CON2*(1.0D0-SQRT(1.0D0-2.0D0*TAU*CON1/CON2**2))
     1              /CON1
              ELSE
                ETAP=TAU/CON2
              ENDIF
              FUN(K)=(PACI(I+1)-PACI(I))
     1              *(1.0D0+(AI(I)+BI(I)*ETAP)*ETAP)**2
     2              /((1.0D0-BI(I)*ETAP*ETAP)*CI*(XTI(I+1)-XTI(I)))
            ENDDO
            CALL SIMPSU(DX,FUN,SUM,NIP)
            PMAX_linear_e(IE) = PACI(I)+SUM(NIP)                     !! OLD: PMAX(IE,M)=PACI(I)+SUM(NIP)    !!DeBuG!!
          ELSE
            PMAX_linear_e(IE) = 1.0D0                                !! OLD:  PMAX(IE,M)=1.0D0
          ENDIF
        ELSE
          PMAX_linear_e(IE) = PACI(1)                                !! OLD:  PMAX(IE,M)=PACI(1)
        ENDIF
      ENDDO

      RETURN
      END
      
    

    
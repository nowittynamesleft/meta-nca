{}:

let
  # Import mach-nix
  #mach-nix = import (fetchTarball "https://github.com/DavHau/mach-nix/tarball/master") {};

  pkgs = import <nixpkgs> {};
  #machNixEnv = mach-nix.mkPython {
  #  requirements = ''auto-gptq'';
  #};

in

pkgs.mkShell {
  buildInputs = [
    # CUDA and related system libraries
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.nccl
    pkgs.linuxPackages.nvidia_x11
    #pkgs.gcc11
    #pkgs.libGLU
    #pkgs.libGL
    #pkgs.xorg.libXi
    #pkgs.xorg.libXmu
    #pkgs.freeglut
    #pkgs.xorg.libXext
    #pkgs.xorg.libX11
    #pkgs.xorg.libXv
    #pkgs.xorg.libXrandr
    #pkgs.zlib
    #pkgs.ncurses5
    #pkgs.stdenv.cc
    #pkgs.binutils
    # the one thing i need mach-nix for
    #machNixEnv

    # more python packages that are nixpkgs based
    pkgs.python3Packages.torchWithCuda
    pkgs.python3Packages.transformers
    pkgs.python3Packages.ipdb
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.wandb
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.imageio
    #pkgs.python3Packages.pip
    #pkgs.python3Packages.virtualenv
   ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit.lib}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.nccl}/lib:$LD_LIBRARY_PATH
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"          
  '';
}

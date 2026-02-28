{
  description = "faster-whisper transcription environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      pythonEnv = pkgs.python3.withPackages (ps: [
        ps.faster-whisper
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          # transcription dependencies
          pythonEnv
          pkgs.ffmpeg
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.libcublas

          # bash script dependencies
          pkgs.unzip
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.zlib}/lib:${pkgs.libgcc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.libcublas}/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}

{
  description = "road-defect: YOLO road damage detection with severity scoring";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      runtimeLibs = with pkgs; [ stdenv.cc.cc.lib zlib libGL glib ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ python312 uv git curl unzip ];
        LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + pkgs.lib.makeLibraryPath runtimeLibs;
        UV_PYTHON_DOWNLOADS = "never";
        shellHook = ''
          if [ ! -x .venv/bin/python ]; then uv venv .venv --python ${pkgs.python312}/bin/python3.12; fi
          source .venv/bin/activate
        '';
      };
    };
}

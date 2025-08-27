class Env {
  static API_USER = import.meta.env.VITE_API_USER || "quantinaut";
  static API_PASS = import.meta.env.VITE_API_PASS || "password";
  static API_HOST = import.meta.env.VITE_API_HOST || "http://localhost:8000";
}

export default Env;

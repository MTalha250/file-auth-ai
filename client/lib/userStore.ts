import { create } from "zustand";

interface UserState {
  access: string | null;
  user: any;
  setAccess: (access: string | null) => void;
  setUser: (user: any) => void;
  logout: () => void;
}

export const useUserStore = create<UserState>((set) => ({
  access: typeof window !== "undefined" ? localStorage.getItem("access") : null,
  user: null,
  setAccess: (access) => {
    set({ access });
    if (typeof window !== "undefined") {
      if (access) localStorage.setItem("access", access);
      else localStorage.removeItem("access");
    }
  },
  setUser: (user) => set({ user }),
  logout: () => {
    set({ access: null, user: null });
    if (typeof window !== "undefined") {
      localStorage.removeItem("access");
      localStorage.removeItem("refresh");
    }
  },
}));

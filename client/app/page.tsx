"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useUserStore } from "../lib/userStore";

export default function Home() {
  const router = useRouter();
  const access =
    typeof window !== "undefined" ? localStorage.getItem("access") : null;

  useEffect(() => {
    if (access) {
      router.replace("/dashboard");
    } else {
      router.replace("/login");
    }
  }, [access, router]);

  return null;
}

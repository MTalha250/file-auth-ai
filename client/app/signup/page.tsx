"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import api from "../../lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Eye,
  EyeOff,
  FileText,
  Check,
  X,
  UserPlus,
  Mail,
  Lock,
  User,
} from "lucide-react";

export default function SignupPage() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState({
    length: false,
    uppercase: false,
    lowercase: false,
    number: false,
    special: false,
  });
  const router = useRouter();

  // Check password strength
  useEffect(() => {
    setPasswordStrength({
      length: password.length >= 8,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      number: /\d/.test(password),
      special: /[!@#$%^&*(),.?":{}|<>]/.test(password),
    });
  }, [password]);

  const getPasswordStrengthScore = () => {
    return Object.values(passwordStrength).filter(Boolean).length;
  };

  const getPasswordStrengthText = () => {
    const score = getPasswordStrengthScore();
    if (score < 2) return { text: "Weak", color: "text-red-500" };
    if (score < 4) return { text: "Medium", color: "text-yellow-500" };
    return { text: "Strong", color: "text-blue-500" };
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    if (password !== passwordConfirm) {
      setError("Passwords do not match");
      setIsLoading(false);
      return;
    }

    if (getPasswordStrengthScore() < 3) {
      setError("Please choose a stronger password");
      setIsLoading(false);
      return;
    }

    try {
      await api.post("/auth/register/", {
        username,
        email,
        password,
        password_confirm: passwordConfirm,
      });
      router.push("/login");
    } catch (err) {
      setError(
        "Registration failed. Please check your information and try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 p-4">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse delay-500"></div>
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo/Brand section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl shadow-lg mb-4 transform hover:scale-105 transition-transform duration-200">
            <FileText className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
            FileAuthAI
          </h1>
          <p className="text-slate-600 mt-2 font-medium">
            Join our secure platform
          </p>
        </div>

        <Card className="backdrop-blur-sm bg-white/80 shadow-2xl border-0 ring-1 ring-slate-200/50">
          <CardHeader className="space-y-1 pb-6">
            <CardTitle className="text-2xl font-semibold text-center text-slate-800">
              Create your account
            </CardTitle>
            <p className="text-sm text-slate-600 text-center">
              Start managing your files securely today
            </p>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <label
                  htmlFor="username"
                  className="text-sm font-medium text-slate-700 flex items-center space-x-2"
                >
                  <User className="w-4 h-4" />
                  <span>Username</span>
                </label>
                <Input
                  id="username"
                  placeholder="Choose a username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="h-11 bg-white border-slate-200 focus:border-blue-500 focus:ring-blue-500/20 transition-all duration-200"
                />
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="email"
                  className="text-sm font-medium text-slate-700 flex items-center space-x-2"
                >
                  <Mail className="w-4 h-4" />
                  <span>Email</span>
                </label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="h-11 bg-white border-slate-200 focus:border-blue-500 focus:ring-blue-500/20 transition-all duration-200"
                />
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="password"
                  className="text-sm font-medium text-slate-700 flex items-center space-x-2"
                >
                  <Lock className="w-4 h-4" />
                  <span>Password</span>
                </label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Create a strong password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="h-11 bg-white border-slate-200 focus:border-blue-500 focus:ring-blue-500/20 transition-all duration-200 pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  >
                    {showPassword ? (
                      <EyeOff className="w-4 h-4" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                  </button>
                </div>

                {/* Password strength indicator */}
                {password && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-600">
                        Password strength:
                      </span>
                      <span
                        className={`text-xs font-medium ${
                          getPasswordStrengthText().color
                        }`}
                      >
                        {getPasswordStrengthText().text}
                      </span>
                    </div>
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5].map((i) => (
                        <div
                          key={i}
                          className={`h-1 flex-1 rounded-full ${
                            i <= getPasswordStrengthScore()
                              ? getPasswordStrengthScore() < 2
                                ? "bg-red-400"
                                : getPasswordStrengthScore() < 4
                                ? "bg-yellow-400"
                                : "bg-blue-400"
                              : "bg-slate-200"
                          }`}
                        />
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      {Object.entries({
                        "8+ characters": passwordStrength.length,
                        Uppercase: passwordStrength.uppercase,
                        Lowercase: passwordStrength.lowercase,
                        Number: passwordStrength.number,
                        "Special char": passwordStrength.special,
                      }).map(([label, met]) => (
                        <div
                          key={label}
                          className="flex items-center space-x-1"
                        >
                          {met ? (
                            <Check className="w-3 h-3 text-blue-500" />
                          ) : (
                            <X className="w-3 h-3 text-slate-300" />
                          )}
                          <span
                            className={met ? "text-blue-700" : "text-slate-500"}
                          >
                            {label}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="passwordConfirm"
                  className="text-sm font-medium text-slate-700 flex items-center space-x-2"
                >
                  <Lock className="w-4 h-4" />
                  <span>Confirm Password</span>
                </label>
                <div className="relative">
                  <Input
                    id="passwordConfirm"
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="Confirm your password"
                    value={passwordConfirm}
                    onChange={(e) => setPasswordConfirm(e.target.value)}
                    required
                    className={`h-11 bg-white border-slate-200 focus:border-blue-500 focus:ring-blue-500/20 transition-all duration-200 pr-10 ${
                      passwordConfirm && password !== passwordConfirm
                        ? "border-red-300"
                        : ""
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="w-4 h-4" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                  </button>
                </div>
                {passwordConfirm && password !== passwordConfirm && (
                  <p className="text-xs text-red-500 flex items-center space-x-1">
                    <X className="w-3 h-3" />
                    <span>Passwords do not match</span>
                  </p>
                )}
                {passwordConfirm && password === passwordConfirm && (
                  <p className="text-xs text-blue-500 flex items-center space-x-1">
                    <Check className="w-3 h-3" />
                    <span>Passwords match</span>
                  </p>
                )}
              </div>

              {error && (
                <Alert
                  variant="destructive"
                  className="bg-red-50 border-red-200 text-red-800"
                >
                  <X className="w-4 h-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <Button
                type="submit"
                onClick={handleSignup}
                className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium rounded-lg shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200"
                disabled={
                  isLoading ||
                  password !== passwordConfirm ||
                  getPasswordStrengthScore() < 3
                }
              >
                {isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Creating account...</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <UserPlus className="w-4 h-4" />
                    <span>Create Account</span>
                  </div>
                )}
              </Button>

              <div className="text-center pt-4 border-t border-slate-100">
                <p className="text-sm text-slate-600">
                  Already have an account?{" "}
                  <a
                    href="/login"
                    className="font-medium text-blue-600 hover:text-blue-700 hover:underline transition-colors duration-200"
                  >
                    Sign in here
                  </a>
                </p>
              </div>
            </div>

            {/* Terms and Privacy */}
            <div className="pt-4 border-t border-slate-100">
              <p className="text-xs text-slate-500 text-center">
                By creating an account, you agree to our{" "}
                <a href="#" className="text-blue-600 hover:underline">
                  Terms of Service
                </a>{" "}
                and{" "}
                <a href="#" className="text-blue-600 hover:underline">
                  Privacy Policy
                </a>
              </p>
            </div>

            {/* Security features */}
            <div className="pt-4 border-t border-slate-100">
              <div className="flex items-center justify-center space-x-6 text-xs text-slate-500">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span>Encrypted</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span>Private</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>Secure</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center mt-8 text-xs text-slate-500">
          <p>© 2025 FileAuthAI. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}

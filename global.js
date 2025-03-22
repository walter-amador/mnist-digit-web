const burgerBtn = document.querySelector('.burger');
const navigation = document.querySelector('.navigation');

burgerBtn.addEventListener('click', () => {
  navigation.classList.toggle('active');
});
